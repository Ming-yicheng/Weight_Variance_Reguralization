#adversarial_samples_generation.py
import os
import os.path as osp
import argparse
import torch
import numpy as np
import torch.nn.functional as F

from torchvision import transforms

from dataset.cub200 import CUB200Data
from dataset.mit67 import MIT67Data
from dataset.stanford_dog import SDog120Data
from dataset.stanford_40 import Stanford40Data
from dataset.flower102 import Flower102Data

from model.fe_resnet import feresnet18, feresnet34, feresnet50, feresnet101

from adversary_factory import get_adversary

def myloss(yhat, y):

    beta = 0.1
    target_loss_part = (yhat - y)**2
    # 权重: 目标神经元权重为1, 其他为beta
    weights = torch.ones_like(yhat)
    weights[y == 0] = beta
    
    weighted_loss = target_loss_part * weights
    # 返回负的平均损失，因为攻击器会最大化这个值
    return -weighted_loss.mean()

class FeatureDestructionLoss:
    def __init__(self, model, source_model_name, target_layer=None):
        self.model = model
        self.target_features = None
        self.current_features = None
        
        # 自动定位层级
        if target_layer is None:
            if 'resnet' in source_model_name:
                self.layer = model.layer4  # 建议从 layer2 开始，迁移性比 layer1 更好
            elif 'mobilenet' in source_model_name:
                self.layer = model.features[7] # 中间层特征
        else:
            self.layer = target_layer

        # 注册 Hook
        self.handle = self.layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # 确保在攻击迭代中保持计算图开启
        self.current_features = output

    def set_clean_reference(self, x):
        """在扰动前，先获取干净样本的基准特征"""
        with torch.no_grad():
            self.model(x)
            self.target_features = self.current_features.detach().clone()

    def __call__(self, yhat, y):
        """
        yhat 和 y 是 advertorch 默认传进来的模型输出和标签，此处我们忽略它们，
        直接使用 Hook 抓取的特征层进行对比。
        """
        if self.target_features is None:
            return torch.tensor(0.0).to(yhat.device)

        adv_feat = self.current_features
        clean_feat = self.target_features

        # 1. 计算余弦相似度 (维度展平: [B, C, H, W] -> [B, D])
        adv_flat = adv_feat.reshape(adv_feat.size(0), -1)
        clean_flat = clean_feat.reshape(clean_feat.size(0), -1)
        
        # cos_sim 范围 [-1, 1]，1 表示方向完全一致
        cos_sim = F.cosine_similarity(adv_flat, clean_flat, dim=1).mean()


        # 攻击目标是让相似度越小越好（方向偏离），距离越大越好。
        # 由于 advertorch 是最大化 loss，所以我们返回负的相似度
        return -cos_sim

    def remove(self):
        self.handle.remove()


def get_args():
    parser = argparse.ArgumentParser(description="Generate Target-Agnostic Adversarial Samples")
    parser.add_argument("--datapath", type=str, default='/data', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='CUB200Data',help='Target dataset. {SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data}')
    parser.add_argument("--source_model", type=str, default='resnet18',help='Source model. {resnet18, resnet50, mobilenet_v2}')
    parser.add_argument("--attack_method", type=str, default='pgd', choices=['pgd', 'mim', 'tifgsm', 'difgsm'],help='Adversarial attack method.')
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget (eps)')
    parser.add_argument("--attack_iter", type=int, default=40, help='Number of attack iterations.')

    # --- 论文特定的参数 ---
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for target activation value')
    parser.add_argument("--neuron_idx", type=int, default=0, help="Index of the neuron to attack.")

    # --- TIFGSM/DIFGSM 参数 ---
    parser.add_argument("--kernel_size", type=int, default=5, help='Gaussian kernel size for TIFGSM.')
    parser.add_argument("--resize_rate", type=float, default=0.9, help='Resize rate for DIFGSM.')
    parser.add_argument("--diversity_prob", type=float, default=0.5, help='Probability for DIFGSM.')

    # --- 其他参数 ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default='adv_samples', help='Directory to save adversarial examples')

    return parser.parse_args()

def main():
    args = get_args()
    print(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_set = eval(args.dataset)(
        args.datapath, False, transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), normalize,
        ]), -1, 98, preload=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )
    
    print(f"Loading source model: {args.source_model}")
    pretrained_model = eval('fe{}'.format(args.source_model))(pretrained=True).eval().cuda()

    # adversary = get_adversary(pretrained_model, myloss, args)

    all_adv_samples, all_labels = [], []

    # # 获取特征数量
    # if 'resnet' in args.source_model:
    #     num_features = pretrained_model.fc.in_features
    # elif 'mobilenet' in args.source_model:
    #     num_features = pretrained_model.classifier[1].in_features

    criterion = FeatureDestructionLoss(pretrained_model, args.source_model)
    adversary = get_adversary(pretrained_model, criterion, args)

    for i, (batch, label) in enumerate(test_loader):
        batch, label = batch.to('cuda'), label.to('cuda')
        
        #准备目标向量 y
        # y = torch.zeros(batch.shape[0], num_features, device='cuda')
        # y[:, args.neuron_idx] = args.m
        
        criterion.set_clean_reference(batch)
        # 这里的 dummy_y 只是为了占位，criterion 内部不会用到它
        dummy_y = torch.zeros(batch.shape[0]).cuda()

        # adv_batch = adversary.perturb(batch, y)
        adv_batch = adversary.perturb(batch, dummy_y)
        
        all_adv_samples.append(adv_batch.cpu())
        all_labels.append(label.cpu())
        
        print(f'Generated batch {i+1}/{len(test_loader)}')

    criterion.remove()

    adv_samples = torch.cat(all_adv_samples, dim=0)
    labels = torch.cat(all_labels, dim=0)

    save_filename = (f'{args.source_model}_{args.dataset}_{args.attack_method.upper()}_'
                    f'B{args.B}_m1000.0_early_layer.pth')
    save_path = osp.join(args.output_dir, save_filename)

    torch.save({
        'adv_samples': adv_samples,
        'labels': labels
    }, save_path)

    print(f'Adversarial samples saved to: {save_path}')
    print(f'Total samples: {len(adv_samples)}')

if __name__ == '__main__':
    main()