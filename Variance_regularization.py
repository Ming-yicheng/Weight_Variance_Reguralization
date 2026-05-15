import os
import os.path as osp
import time
from pdb import set_trace as st

import torch
import numpy as np
import pandas as pd
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
from model.vgg import vgg16_bn_dropout
from model.vgg import fevgg16_bn

from eval_robustness import advtest, myloss
from utils import *
from finetuner import Finetuner

class Variance_regularization(Finetuner):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(Variance_regularization, self).__init__(
            args, model, teacher, train_loader, test_loader
        )
        self.outlier_layers_to_constrain = self._get_outlier_layers_by_variance()

    def _get_outlier_layers_by_variance(self, iqr_multiplier: float = 1.5) -> set:

        print("--- Analyzing teacher model to find outlier layers ---")

        layer_variances = {}
        for name, module in self.teacher.named_modules():
            if isinstance(module, nn.Conv2d):
                weights = module.weight.data
                if weights.numel() > 1:
                    layer_variances[name] = torch.var(weights).item()

        if not layer_variances:
            return set()

        # 2. IQR 分析
        df = pd.DataFrame(list(layer_variances.items()), columns=['Layer', 'Variance'])
        Q1 = df['Variance'].quantile(0.25)
        Q3 = df['Variance'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + iqr_multiplier * IQR
        
        outlier_df = df[df['Variance'] > outlier_threshold]
        outlier_layer_names = set(outlier_df['Layer'].tolist())
        
        print(f"Found {len(outlier_layer_names)} outlier layers: {outlier_layer_names}")
        print("--- Analysis complete ---")
        
        return outlier_layer_names

    def train(self, ):
        model = self.model
        train_loader = self.train_loader
        iterations = self.args.iterations
        lr = self.args.lr
        output_dir = self.args.output_dir
        teacher = self.teacher
        args = self.args
        model = model.to('cuda')
        
        if 'resnet' in args.network:
            fc_module = self.model.fc
        elif 'vgg' in args.network or 'mobilenet' in args.network:
            fc_module = self.model.classifier
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

        # WVR MODIFICATION: 为WVR损失添加新的监控器(Meter)
        batch_time = MovingAverageMeter('Time', ':6.3f')
        data_time = MovingAverageMeter('Data', ':6.3f')
        ce_loss_meter = MovingAverageMeter('CE Loss', ':6.3f')
        wvr_loss_meter = MovingAverageMeter('WVR Loss', ':.6e') # 新增
        top1_meter  = MovingAverageMeter('Acc@1', ':6.2f')

        test_path = osp.join(output_dir, "test.tsv")
        with open(test_path, 'w') as wf:
            columns = ['time', 'iter', 'Acc', 'celoss']
            wf.write('\t'.join(columns) + '\n')
        # adv_path = osp.join(output_dir, "adv.tsv")
        # with open(adv_path, 'w') as wf:
        #     columns = ['time', 'iter', 'Acc', 'AdvAcc', 'ASR']
        #     wf.write('\t'.join(columns) + '\n')
        
        dataloader_iterator = iter(train_loader)
        for i in range(iterations):
            model.train()
            
            end = time.time()
            try:
                batch, label = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                batch, label = next(dataloader_iterator)
            batch, label = batch.to('cuda'), label.to('cuda')
            data_time.update(time.time() - end)
            
            # --- WVR MODIFICATION: 核心训练逻辑 ---
            # 1. 计算原始的CE Loss
            out = model(batch)
            ce_loss = ce(out, label)

            # 2. 计算分层权重方差损失 (WVR Loss)
            wvr_loss = torch.tensor(0.0, device='cuda')
            current_wvr_lambda = args.wvr_lambda

            if current_wvr_lambda != 0:
                # 遍历模型中的所有卷积层模块
                for module in model.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        weight = module.weight
                        
                        if weight.requires_grad and weight.numel() > 1:
                            # 重塑权重: [out_channels, in_channels, kernel_h, kernel_w] -> [out_channels, -1]
                            weight_reshaped = weight.view(weight.size(0), -1)
                            
                            # 计算每个卷积核的方差 (沿dim=1计算)
                            kernel_vars = torch.var(weight_reshaped, dim=1)
                            
                            # 排除无效的方差值并求平均
                            valid_vars = kernel_vars[kernel_vars.isfinite()]
                            if len(valid_vars) > 0:
                                wvr_loss += torch.mean(valid_vars)

            # 3. 计算总损失
            loss = ce_loss + current_wvr_lambda * wvr_loss

            # 4. 计算准确率 (Top-1)
            _, pred = out.max(dim=1)
            top1 = float(pred.eq(label).sum().item()) / label.shape[0] * 100.
            # --- WVR MODIFICATION END ---

            # 更新监控器
            top1_meter.update(top1)
            ce_loss_meter.update(ce_loss.item())
            wvr_loss_meter.update(wvr_loss.item()) 
            
            # 在反向传播前清零梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)

            if (i % args.print_freq == 0) or (i == iterations-1):
                # 将 wvr_loss_meter 加入到进度打印中
                progress = ProgressMeter(
                    iterations,
                    [batch_time, data_time, top1_meter, ce_loss_meter, wvr_loss_meter],
                    prefix="PID {} ".format(self.args.pid),
                    output_dir=output_dir,
                )
                progress.display(i)

            if (i % args.test_interval == 0) or (i == iterations-1):
                test_top1, test_ce_loss = self.test()
                print(
                    'Eval Test | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | PID {}'.format(i+1, iterations, test_top1, test_ce_loss, self.args.pid))
                localtime = time.asctime( time.localtime(time.time()) )[4:-6]
                with open(test_path, 'a') as af:
                    test_cols = [
                        localtime,
                        i, 
                        round(test_top1,2), 
                        round(test_ce_loss,2), 
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

            # if ( 
            #     args.adv_test_interval > 0 and 
            #     ( (i % args.adv_test_interval == 0) or (i == iterations-1) )
            # ):
            #     clean_top1, adv_top1, adv_sr = self.adv_eval_fn(model)
            #     localtime = time.asctime( time.localtime(time.time()) )[4:-6]
            #     with open(adv_path, 'a') as af:
            #         test_cols = [
            #             localtime,
            #             i, 
            #             round(clean_top1,2),
            #             round(adv_top1,2),
            #             round(adv_sr,2),
            #         ]
            #         af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        return model