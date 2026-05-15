import torch
import numpy as np
import random
from torchvision.datasets import CIFAR100
from collections import defaultdict

class CIFAR100Data(torch.utils.data.Dataset):
    """
    CIFAR-100 dataset handler that supports subsetting by class and data fraction/fixed size.
    """
    def __init__(
        self, 
        root, 
        is_train=True, 
        transform=None, 
        num_classes_subset=-1,
        data_fraction=1.0,
        fixed_total_size=None,
        seed=42,
        download=True
    ):
        """
        Args:
            root (str): Path to the dataset directory.
            is_train (bool): Whether to load the training or test set.
            transform (callable, optional): A function/transform to apply to the images.
            num_classes_subset (int): Use the first N classes. -1 for all 100 classes.
            data_fraction (float): [Experiment 2] Fraction of data to use for each class (0.0 to 1.0).
            fixed_total_size (int): [Experiment 1] Fix the total number of training samples.
            seed (int): Random seed for reproducibility of sampling.
            download (bool): Whether to download the dataset if not found.
        """
        self.root = root
        self.is_train = is_train
        self.transform = transform
        
        # 使用 torchvision 加载原始数据集
        base_dataset = CIFAR100(root=self.root, train=self.is_train, download=download)
        
        # --- 数据和标签预处理 ---
        data = torch.from_numpy(base_dataset.data).permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        targets = torch.tensor(base_dataset.targets)
        
        # --- 核心逻辑: 数据子集划分 ---
        
        # 1. 根据类别数量筛选 (num_classes_subset)
        if num_classes_subset == -1:
            num_classes_subset = 100 # -1 表示使用所有类别
        
        # 找出属于前 N 个类别的样本的索引
        class_indices_mask = targets < num_classes_subset
        data = data[class_indices_mask]
        targets = targets[class_indices_mask]
        
        self.num_classes = num_classes_subset
        
        # 2. 如果是训练集，则进行数据量采样 (data_fraction 或 fixed_total_size)
        if self.is_train:
            # 按类别组织样本索引
            class_to_indices = defaultdict(list)
            for i, target in enumerate(targets):
                class_to_indices[target.item()].append(i)
            
            final_indices = []
            
            # 设置随机种子以保证采样可复现
            random.seed(seed)
            np.random.seed(seed)
            
            # 实验一: 固定总样本数
            if fixed_total_size is not None:
                if self.num_classes == 0:
                    raise ValueError("Cannot use fixed_total_size with 0 classes.")
                
                # 计算每个类别应该分配多少样本
                samples_per_class = fixed_total_size // self.num_classes
                
                print(f"[Dataset] Experiment 1 Mode: Fixing total size to {fixed_total_size}.")
                print(f"[Dataset] Each of the {self.num_classes} classes will have {samples_per_class} samples.")
                
                for class_idx in range(self.num_classes):
                    indices = class_to_indices[class_idx]
                    # 如果请求的样本数超过了该类别的总样本数，则发出警告并使用所有样本
                    if samples_per_class > len(indices):
                        print(f"Warning: Requested {samples_per_class} samples for class {class_idx}, but only {len(indices)} are available. Using all available samples.")
                        final_indices.extend(indices)
                    else:
                        final_indices.extend(random.sample(indices, samples_per_class))

            # 实验二: 按比例缩减样本数
            elif data_fraction < 1.0:
                print(f"[Dataset] Experiment 2 Mode: Using {data_fraction*100:.1f}% of data for each class.")
                for class_idx in range(self.num_classes):
                    indices = class_to_indices[class_idx]
                    num_samples_to_keep = int(len(indices) * data_fraction)
                    final_indices.extend(random.sample(indices, num_samples_to_keep))
            
            # 标准模式: 使用所有筛选后的数据
            else:
                final_indices = list(range(len(targets)))
            
            # 使用最终的索引列表来选择数据和标签
            self.data = data[final_indices]
            self.targets = targets[final_indices]

        else: # 如果是测试集，则不进行采样，仅根据类别数筛选
            self.data = data
            self.targets = targets
            
        print(f"[Dataset] {'Train' if self.is_train else 'Test'} set loaded. Samples: {len(self.data)}, Classes: {self.num_classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # torchvision transform 期望 PIL Image 或 (H, W, C) numpy array
        # 我们需要将 (C, H, W) tensor 转回 (H, W, C) numpy
        img = img.permute(1, 2, 0).numpy()
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target