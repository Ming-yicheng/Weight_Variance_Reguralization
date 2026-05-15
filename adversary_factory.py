# adversary_factory.py

import torch
import torch.nn.functional as F
import numpy as np

# 导入 advertorch 库中的攻击方法和基类
from advertorch.attacks import (
    LinfPGDAttack,
    LinfMomentumIterativeAttack
)
from advertorch.attacks.base import Attack, LabelMixin
from advertorch.utils import clamp


def get_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3):
    """为TIFGSM生成高斯核"""
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1) / 2.
    variance = sigma**2.
    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                    torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    return gaussian_kernel

class TI(LinfMomentumIterativeAttack):
    """
    TIFGSM: Translation-Invariant Attack
    在每次迭代中，使用高斯核对梯度进行卷积，以平滑梯度，提高攻击的迁移性。
    """
    def __init__(self, predict, loss_fn, eps, nb_iter, eps_iter, decay_factor=1.0,
                 kernel_size=5, sigma=1.0, clip_min=0., clip_max=1., targeted=False):
        super(TI, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, decay_factor,
            clip_min, clip_max, targeted)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = get_gaussian_kernel(kernel_size, sigma, 3)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)
        
        delta.requires_grad_()

        for i in range(self.nb_iter):
            if delta.grad is not None:
                delta.grad.data.zero_()

            img_adv = x + delta
            outputs = self.predict(img_adv)
            loss = self.loss_fn(outputs, y)
            if self.targeted:
                loss = -loss
            loss.backward()

            grad = delta.grad.data
            if self.gaussian_kernel.device != grad.device:
                self.gaussian_kernel = self.gaussian_kernel.to(grad.device)
            # 核心：卷积梯度
            grad = F.conv2d(grad, self.gaussian_kernel, padding='same', groups=3)
            
            g = self.decay_factor * g + grad / torch.norm(grad, p=1, keepdim=True)
            
            delta.data += self.eps_iter * torch.sign(g)
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x
        
        return x + delta.data

class DI(LinfMomentumIterativeAttack):
    """
    DIFGSM: Diverse Inputs Attack
    在每次迭代计算梯度时，以一定概率对输入进行随机缩放和填充，以提高迁移性。
    """
    def __init__(self, predict, loss_fn, eps, nb_iter, eps_iter, decay_factor=1.0,
                 resize_rate=0.9, diversity_prob=0.5,
                 clip_min=0., clip_max=1., targeted=False):
        super(DI, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, decay_factor,
            clip_min, clip_max, targeted)
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def input_diversity(self, x):
        if torch.rand(1).item() > self.diversity_prob:
            return x
        
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        
        if img_resize < img_size:
            rnd = torch.randint(img_size - img_resize, size=(1,)).item()
            x_resized = F.interpolate(x, size=[img_resize, img_resize], mode='bilinear', align_corners=False)
            x_padded = F.pad(x_resized, [rnd, img_size - img_resize - rnd, rnd, img_size - img_resize - rnd], mode='constant', value=0)
            return x_padded
        else:
            return x

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)

        delta.requires_grad_()

        for i in range(self.nb_iter):
            if delta.grad is not None:
                delta.grad.data.zero_()
            
            # 核心：对输入进行变换
            img_adv_div = self.input_diversity(x + delta)
            
            outputs = self.predict(img_adv_div)
            loss = self.loss_fn(outputs, y)
            if self.targeted:
                loss = -loss
            loss.backward()

            grad = delta.grad.data
            g = self.decay_factor * g + grad / torch.norm(grad, p=1, keepdim=True)
            
            delta.data += self.eps_iter * torch.sign(g)
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = clamp(x + delta.data, min=self.clip_min, max=self.clip_max) - x
            
        return x + delta.data



def get_adversary(model, loss_fn, args):
    """
    根据传入的参数args，创建并返回一个配置好的对抗攻击器实例。
    
    :param model: 要攻击的模型。
    :param loss_fn: 攻击时使用的损失函数。
    :param args: 命令行解析后的参数对象。
    :return: 一个配置好的攻击器实例。
    """
    print(f"Initializing attack: {args.attack_method.upper()}")
    
    # 标准化后的数据范围，根据 ImageNet 预训练模型的通常做法
    clip_min, clip_max = -2.2, 2.2 

    if args.attack_method == 'pgd':
        return LinfPGDAttack(
            model, loss_fn=loss_fn, eps=args.B,
            nb_iter=args.attack_iter, eps_iter=args.B/10,
            rand_init=True, clip_min=clip_min, clip_max=clip_max,
            targeted=False)
    
    elif args.attack_method == 'mim':
        return LinfMomentumIterativeAttack(
            model, loss_fn=loss_fn, eps=args.B,
            nb_iter=args.attack_iter, eps_iter=args.B/10,
            decay_factor=1.0, clip_min=clip_min, clip_max=clip_max,
            targeted=False)

    elif args.attack_method == 'tifgsm':
        return TI(
            model, loss_fn=loss_fn, eps=args.B,
            nb_iter=args.attack_iter, eps_iter=args.B/10,
            kernel_size=args.kernel_size, clip_min=clip_min, clip_max=clip_max,
            targeted=False)

    elif args.attack_method == 'difgsm':
        return DI(
            model, loss_fn=loss_fn, eps=args.B,
            nb_iter=args.attack_iter, eps_iter=args.B/10,
            resize_rate=args.resize_rate, diversity_prob=args.diversity_prob,
            clip_min=clip_min, clip_max=clip_max,
            targeted=False)
    else:
        raise NotImplementedError(f"Attack method '{args.attack_method}' not implemented.")