import os
import os.path as osp
import time
from pdb import set_trace as st

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from advertorch.attacks import LinfPGDAttack

from dataset.cub200 import CUB200Data
from dataset.mit67 import MIT67Data
from dataset.stanford_dog import SDog120Data
from dataset.stanford_40 import Stanford40Data
from dataset.flower102 import Flower102Data

from model.fe_resnet import resnet18_dropout, resnet34_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet34, feresnet50, feresnet101

from eval_robustness import advtest, myloss
from utils import *
from finetuner import Finetuner

class IPLA(Finetuner):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(IPLA, self).__init__(
            args, model, teacher, train_loader, test_loader
        )
        
        print("进入IPLA")
        self.log_path = osp.join(self.args.output_dir, "prune.log")
        with open(self.log_path, "w") as f:
            f.write("--- Pruning Log Initialized ---\n\n")
        num_steps = self.args.num_steps 
        weights_prune_ratio = self.args.weights_prune_ratio

        for i in range(num_steps):
            current_w_ratio = 1 - (1-weights_prune_ratio) ** (i+1)
            self.zero_prune_weights(current_w_ratio)
            self.train(i)

    def prune_record(self, log: str):
        with open(self.log_path, 'a') as logger:
            logger.write(log + "\n")

    def LAMP_scores(self, scores):
        sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
        scores_cumsum_temp = sorted_scores.cumsum(dim=0)
        scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
        scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
        sorted_scores /= (scores.sum() - scores_cumsum)
        new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
        new_scores[sorted_idx] = sorted_scores
        return new_scores.view(scores.shape)
    
    def zero_prune_weights(self, prune_ratio=0.2):
        """
        根据权重绝对值的LAMP分数进行全局剪枝。
        """
        # 1. 收集所有待剪枝层的权重绝对值，并计算其 LAMP 分数
        lamp_scores = {}
        all_scores = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and "downsample" not in name:
                weight_abs = torch.abs(module.weight.data)
                layer_scores = self.LAMP_scores(weight_abs)
                lamp_scores[name] = layer_scores
                all_scores.append(layer_scores.view(-1))
        
        if not all_scores:
            print("没有可供剪枝的层。")
            return

        all_scores_tensor = torch.cat(all_scores)
        total_weights = all_scores_tensor.numel()

        # 2. 根据剪枝率计算全局阈值
        num_weights_to_prune = int(total_weights * prune_ratio)
        
        if num_weights_to_prune == 0:
            print(f"剪枝率 {prune_ratio:.2%} 过低，在当前步骤中无需剪枝。")
            return
        
        if num_weights_to_prune >= total_weights: # 防止剪掉所有权重
            threshold = all_scores_tensor.max() 
        else:
            threshold = torch.kthvalue(all_scores_tensor, num_weights_to_prune).values

        # 3. 执行剪枝并统计每层信息
        per_layer_prune_count = {}
        final_nonzero_counts = {}

        for name, module in self.model.named_modules():
            if name in lamp_scores:
                # 根据全局阈值生成剪枝掩码
                prune_mask = lamp_scores[name] <= threshold
                
                # 统计本轮被剪掉的数量
                per_layer_prune_count[name] = torch.sum(prune_mask).item()
                
                # 应用掩码，执行剪枝
                module.weight.data[prune_mask] = 0.0
                
                # 统计剪枝后剩余的数量
                final_nonzero_counts[name] = torch.count_nonzero(module.weight.data).item()

        # 4. 打印全局和分层日志
        total_pruned_this_step = sum(per_layer_prune_count.values())
        final_total_nonzero = sum(final_nonzero_counts.values())
        actual_sparsity = 1 - (final_total_nonzero / total_weights)

        log = (
            f"  - 全局剪枝阈值 (LAMP Score): {threshold.item():.4e}\n"
            f"  - 本轮剪枝权重数: {total_pruned_this_step} / {total_weights}\n"
            f"  - 模型当前总稀疏度: {actual_sparsity:.2%}\n"
            f"各层剪枝详情:\n"
        )

        for name, module in self.model.named_modules():
            if name in lamp_scores:
                total_w = module.weight.numel()
                pruned_w = per_layer_prune_count.get(name, 0)
                final_nz = final_nonzero_counts.get(name, 0)
                layer_sparsity = 1 - (final_nz / total_w) if total_w > 0 else 0
                
                log += (
                    f"  - {name:<35} "
                    f"剪枝数: {pruned_w:>8d}/{total_w:<8d} | "
                    f"当前层稀疏度: {layer_sparsity:.2%}\n"
                )

        self.prune_record(log)

    def train(self, prune_step):
        model = self.model
        train_loader = self.train_loader
        iterations = self.args.iterations
        lr = self.args.lr
        output_dir = self.args.output_dir
        teacher = self.teacher
        args = self.args
        model = model.to('cuda')
        
        fc_module = model.fc
        ignored_params = list(map(id, fc_module.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                        self.model.parameters())
        optimizer = torch.optim.SGD(
            [
                {'params': base_params},
                {'params': fc_module.parameters(), 'lr': lr*10}
            ], 
            lr=lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        teacher.eval()
        ce = CrossEntropyLabelSmooth(train_loader.dataset.num_classes)

        batch_time = MovingAverageMeter('Time', ':6.3f')
        data_time = MovingAverageMeter('Data', ':6.3f')
        ce_loss_meter = MovingAverageMeter('CE Loss', ':6.3f')
        top1_meter  = MovingAverageMeter('Acc@1', ':6.2f')

        test_path = osp.join(output_dir, f"test_iter{prune_step}.tsv")
        adv_path = osp.join(output_dir, f"adv_iter{prune_step}.tsv")
        with open(test_path, 'w') as wf:
            columns = ['time', 'iter', 'Acc', 'celoss', 'W_Variance']
            wf.write('\t'.join(columns) + '\n')
        with open(adv_path, 'w') as wf:
            columns = ['time', 'iter', 'Acc', 'AdvAcc', 'ASR']
            wf.write('\t'.join(columns) + '\n')
        
        dataloader_iterator = iter(train_loader)
        for i in range(iterations):
            model.train()
            optimizer.zero_grad()

            end = time.time()
            try:
                batch, label = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                batch, label = next(dataloader_iterator)
            batch, label = batch.to('cuda'), label.to('cuda')
            data_time.update(time.time() - end)

            loss, top1 = self.compute_loss(
                batch, label, ce, 
            )

            top1_meter.update(top1)
            ce_loss_meter.update(loss)
            
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)

            if (i % args.print_freq == 0) or (i == iterations-1):
                progress = ProgressMeter(
                    iterations,
                    [batch_time, data_time, top1_meter, ce_loss_meter],
                    prefix="PID {} ".format(self.args.pid),
                    output_dir=output_dir,
                )
                progress.display(i)

            if (i % args.test_interval == 0) or (i == iterations-1):
                test_top1, test_ce_loss = self.test()

                total_weight_variance = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and 'weight' in name:
                        param_data = param.detach()
                        non_zero_weights = param_data[param_data != 0]
                        if non_zero_weights.numel() > 1:
                            # 对非零权重计算方差，并累加到总和中
                            total_weight_variance += torch.var(non_zero_weights).item()

                print(
                    'Eval Test | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | W_Var: {:.4e} | PID {}'.format(
                        i+1, iterations, test_top1, test_ce_loss, total_weight_variance, self.args.pid
                    )
                )
                
                localtime = time.asctime( time.localtime(time.time()) )[4:-6]
                with open(test_path, 'a') as af:
                    test_cols = [
                        localtime,
                        i, 
                        round(test_top1, 2), 
                        round(test_ce_loss, 2), 
                        f"{total_weight_variance:.4e}"  
                    ]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

                ckpt_path = osp.join(
                    args.output_dir,
                    "ckpt.pth"
                )
                torch.save(
                    {'state_dict': model.state_dict()}, 
                    ckpt_path,
                )

            if ( 
                args.adv_test_interval > 0 and 
                ( (i % args.adv_test_interval == 0) or (i == iterations-1) )
            ):
                clean_top1, adv_top1, adv_sr = self.adv_eval_fn(model)
                localtime = time.asctime( time.localtime(time.time()) )[4:-6]
                with open(adv_path, 'a') as af:
                    test_cols = [
                        localtime,
                        i, 
                        round(clean_top1,2),
                        round(adv_top1,2),
                        round(adv_sr,2),
                    ]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        return model

