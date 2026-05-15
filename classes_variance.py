import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

# --- 依赖和模型部分 (保持不变) ---
from dataset.cifar100_classes import CIFAR100Data # <--- MODIFIED: 导入新的数据集类
from model.fe_resnet import resnet18_dropout, resnet34_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet34, feresnet50, feresnet101
from utils import *
from finetuner import Finetuner

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='./data', help='path to the dataset') # <--- MODIFIED: 修改默认路径
    parser.add_argument("--dataset", type=str, default='CIFAR100Data', help='Target dataset.') # <--- MODIFIED: 修改默认数据集
    parser.add_argument("--iterations", type=int, default=10000, help='Iterations to train')
    parser.add_argument("--print_freq", type=int, default=100, help='Frequency of printing training logs')
    parser.add_argument("--test_interval", type=int, default=1000, help='Frequency of testing')
    parser.add_argument("--network", type=str, default='resnet18', help='Network architecture.')
    parser.add_argument("--name", type=str, default='cifar100_experiment', help='Name for the checkpoint') # <--- MODIFIED: 修改默认实验名
    parser.add_argument("--batch_size", type=int, default=128) # <--- MODIFIED: CIFAR-100 通常用稍大的 batch_size
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4) # <--- MODIFIED: 为 CIFAR-100 设置更常规的权重衰减
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.1, help='Dropout rate for spatial dropout')
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument("--log", action='store_true', default=False, help='Redirect the output to log/args.name.log')
    parser.add_argument("--resume", type=str, default=None, help='指定检查点文件的路径以继续训练')
    # --- 为两个实验新增的参数 (保持不变) ---
    parser.add_argument("--num_classes_subset", type=int, default=100, help='Use the first N classes. -1 for all classes.') # <--- MODIFIED: 默认100个类
    
    # 用于实验一: 固定总数据量
    parser.add_argument("--fixed_total_size", type=int, default=None, help='[Experiment 1] Fix the total number of training samples to this value.')
    
    # 用于实验二: 固定类别数, 改变数据量
    parser.add_argument("--data_fraction", type=float, default=1.0, help='[Experiment 2] Use a fraction of training data for each class.')

    args = parser.parse_args()
    
    # --- 参数校验 (保持不变) ---
    if args.data_fraction != 1.0 and args.fixed_total_size is not None:
        print("Error: --data_fraction and --fixed_total_size cannot be used at the same time. Please choose one experiment type.")
        sys.exit(1)
    
    args.pid = os.getpid()
    args.output_dir = osp.join(
        args.output_dir,
        args.name
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    return args

if __name__=="__main__":
    seed = 98
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = get_args()
    
    print(f"Arguments: {args}")

    # <--- MODIFIED: 使用 CIFAR-100 的标准化参数 ---
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

    # 如果 num_classes_subset 为-1, 则使用全部100个类
    num_classes_for_exp = 100 if args.num_classes_subset == -1 else args.num_classes_subset

    print(f"\n--- Experiment Setup ---")
    print(f"Number of classes: {num_classes_for_exp}")
    if args.fixed_total_size:
        print(f"Mode: Fixed Total Size (Exp 1), Target size: {args.fixed_total_size}")
    elif args.data_fraction != 1.0:
        print(f"Mode: Data Fraction (Exp 2), Fraction: {args.data_fraction*100:.1f}%")
    else:
        print("Mode: Standard Full Training")
    print("------------------------\n")

    # <--- MODIFIED: 核心修改, 加载 CIFAR100Data ---
    # 注意: CIFAR-100 原始图像是32x32。为了使用预训练的ResNet，需要放大到224x224。
    # 训练集增加数据增强
    train_set = CIFAR100Data(
        args.datapath, 
        is_train=True, 
        transform=transforms.Compose([
            transforms.ToPILImage(), # <--- MODIFIED: 先转为PIL Image以适配后续transform
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), 
        num_classes_subset=args.num_classes_subset, 
        data_fraction=args.data_fraction,
        fixed_total_size=args.fixed_total_size,
        seed=seed
    )
    
    # 加载测试集
    test_set = CIFAR100Data(
        args.datapath, 
        is_train=False, 
        transform=transforms.Compose([
            transforms.ToPILImage(), # <--- MODIFIED
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), 
        num_classes_subset=args.num_classes_subset, 
        seed=seed
    )

    if len(train_set) == 0 or len(test_set) == 0:
        print(f"Error: No data found for {num_classes_for_exp} classes. Please check your configuration.")
        sys.exit(1)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    model = eval('{}_dropout'.format(args.network))(
        pretrained=False, 
        dropout=args.dropout, 
        num_classes=num_classes_for_exp 
    )

    teacher = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=0, 
        num_classes=num_classes_for_exp
    )

    finetune_machine = Finetuner(
        args,
        model, teacher,
        train_loader, test_loader,
    )

    finetune_machine.train()