import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module, # 原始VGG的卷积层部分
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.1,  # 用于前两层FC层以及最终分类层
        has_classifier: bool = True # 控制是否包含最终的分类头 (fc8)
    ) -> None:
        super().__init__()
        
        # 特征提取器: 包含原始 VGG 的卷积层、avgpool 层，以及分类器中的前两层FC层
        self.features_extractor = nn.Sequential(
            features,  # 卷积层部分
            nn.AdaptiveAvgPool2d((7, 7)), # 平均池化层
            nn.Flatten(), # 在FC层之前展平
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout)
        )
        self.has_classifier = has_classifier

        if self.has_classifier:
            # 分类器: 只包含最终的分类层 (fc8)
            self.classifier = nn.Sequential(
                nn.Linear(4096, num_classes),
            )
        else:
            # 如果没有分类器，设置为一个 Identity 模块
            self.classifier = nn.Identity()

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear): # 对所有线性层进行初始化
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取阶段
        x = self.features_extractor(x)
        
        # 分类阶段 (如果存在)
        if self.has_classifier:
            x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, conv_dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            
            # 在 ReLU 之后添加 Dropout2d 层
            if conv_dropout > 0.0:
                layers += [nn.Dropout2d(p=conv_dropout)]
            
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG11
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG13
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],  # VGG16
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],  # VGG19
}

_model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def _vgg(
    arch: str, cfg_key: str, batch_norm: bool, pretrained: bool, progress: bool,
    dropout: float = 0.5, conv_dropout: float = 0.0, num_classes: int = 1000, has_classifier: bool = True, **kwargs: Any
) -> VGG:
    model = VGG(make_layers(cfgs[cfg_key], batch_norm=batch_norm, conv_dropout=conv_dropout),
                dropout=dropout, num_classes=num_classes, has_classifier=has_classifier, **kwargs)
    
    if pretrained:
        if arch not in _model_urls:
            raise ValueError(f"Pretrained weights for {arch} are not available.")
        state_dict = load_state_dict_from_url(_model_urls[arch], progress=progress)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('features.'):
                # 原始卷积层部分
                new_state_dict['features_extractor.0.' + k[len('features.'):]] = v
            elif k.startswith('classifier.'):
                # 原始分类器层中的前两层 (fc6, fc7) 移到 features_extractor
                # 原始 key: classifier.0.weight, classifier.0.bias (fc6)
                # 原始 key: classifier.3.weight, classifier.3.bias (fc7)
                # 新 key: features_extractor.3.weight, features_extractor.3.bias (fc6)
                # 新 key: features_extractor.6.weight, features_extractor.6.bias (fc7)
                if '0.weight' in k: # fc6 weight
                    new_state_dict['features_extractor.3.weight'] = v
                elif '0.bias' in k: # fc6 bias
                    new_state_dict['features_extractor.3.bias'] = v
                elif '3.weight' in k: # fc7 weight
                    new_state_dict['features_extractor.6.weight'] = v
                elif '3.bias' in k: # fc7 bias
                    new_state_dict['features_extractor.6.bias'] = v
                elif '6.weight' in k and has_classifier and num_classes == 1000: # fc8 weight，只有当有分类器且类别数匹配时才加载
                    new_state_dict['classifier.0.weight'] = v
                elif '6.bias' in k and has_classifier and num_classes == 1000: # fc8 bias，只有当有分类器且类别数匹配时才加载
                    new_state_dict['classifier.0.bias'] = v
        
        # 严格加载，但如果 num_classes 不为 1000 或者 has_classifier 为 False，则不加载 fc8
        if has_classifier and num_classes != 1000:
            # 如果有分类器但类别数不匹配，则只加载特征提取器部分，分类器重新初始化
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded feature extractor weights for {arch}. Final classifier layer is reinitialized for {num_classes} classes.")
        elif not has_classifier:
            # 如果没有分类器，则只加载特征提取器部分
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded feature extractor weights for {arch}. Final classifier layer is removed.")
        else:
            # 完整加载 (包括重新映射后的 fc8)
            model.load_state_dict(new_state_dict, strict=True)
            print(f"Loaded full pretrained weights for {arch}.")

    return model

# ----------- 便捷的工厂函数，带有 Dropout -----------

def vgg11_dropout(*, pretrained: bool = False, progress: bool = True, dropout: float = 0.5, conv_dropout: float = 0.0, num_classes: int = 1000, **kwargs: Any) -> VGG:
    """VGG-11 model with Dropout.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): Dropout probability for intermediate FC layers and final classifier layer. Default: 0.5.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        num_classes (int): Number of classes in the final classifier layer. Default: 1000. Set to a different value for transfer learning.
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg11", "A", False, pretrained, progress, dropout=dropout, conv_dropout=conv_dropout, num_classes=num_classes, has_classifier=True, **kwargs)


def vgg11_bn_dropout(*, pretrained: bool = False, progress: bool = True, dropout: float = 0.5, conv_dropout: float = 0.0, num_classes: int = 1000, **kwargs: Any) -> VGG:
    """VGG-11-BN model with Dropout.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): Dropout probability for intermediate FC layers and final classifier layer. Default: 0.5.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        num_classes (int): Number of classes in the final classifier layer. Default: 1000. Set to a different value for transfer learning.
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, dropout=dropout, conv_dropout=conv_dropout, num_classes=num_classes, has_classifier=True, **kwargs)


def vgg13_dropout(*, pretrained: bool = False, progress: bool = True, dropout: float = 0.5, conv_dropout: float = 0.0, num_classes: int = 1000, **kwargs: Any) -> VGG:
    """VGG-13 model with Dropout.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): Dropout probability for intermediate FC layers and final classifier layer. Default: 0.5.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        num_classes (int): Number of classes in the final classifier layer. Default: 1000. Set to a different value for transfer learning.
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg13", "B", False, pretrained, progress, dropout=dropout, conv_dropout=conv_dropout, num_classes=num_classes, has_classifier=True, **kwargs)


def vgg13_bn_dropout(*, pretrained: bool = False, progress: bool = True, dropout: float = 0.5, conv_dropout: float = 0.0, num_classes: int = 1000, **kwargs: Any) -> VGG:
    """VGG-13-BN model with Dropout.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): Dropout probability for intermediate FC layers and final classifier layer. Default: 0.5.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        num_classes (int): Number of classes in the final classifier layer. Default: 1000. Set to a different value for transfer learning.
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, dropout=dropout, conv_dropout=conv_dropout, num_classes=num_classes, has_classifier=True, **kwargs)


def vgg16_dropout(*, pretrained: bool = False, progress: bool = True, dropout: float = 0.5, conv_dropout: float = 0.0, num_classes: int = 1000, **kwargs: Any) -> VGG:
    """VGG-16 model with Dropout.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): Dropout probability for intermediate FC layers and final classifier layer. Default: 0.5.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        num_classes (int): Number of classes in the final classifier layer. Default: 1000. Set to a different value for transfer learning.
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg16", "D", False, pretrained, progress, dropout=dropout, conv_dropout=conv_dropout, num_classes=num_classes, has_classifier=True, **kwargs)


def vgg16_bn_dropout(*, pretrained: bool = False, progress: bool = True, dropout: float = 0.5, conv_dropout: float = 0.0, num_classes: int = 1000, **kwargs: Any) -> VGG:
    """VGG-16-BN model with Dropout.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): Dropout probability for intermediate FC layers and final classifier layer. Default: 0.5.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        num_classes (int): Number of classes in the final classifier layer. Default: 1000. Set to a different value for transfer learning.
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, dropout=dropout, conv_dropout=conv_dropout, num_classes=num_classes, has_classifier=True, **kwargs)


def vgg19_dropout(*, pretrained: bool = False, progress: bool = True, dropout: float = 0.5, conv_dropout: float = 0.0, num_classes: int = 1000, **kwargs: Any) -> VGG:
    """VGG-19 model with Dropout.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): Dropout probability for intermediate FC layers and final classifier layer. Default: 0.5.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        num_classes (int): Number of classes in the final classifier layer. Default: 1000. Set to a different value for transfer learning.
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg19", "E", False, pretrained, progress, dropout=dropout, conv_dropout=conv_dropout, num_classes=num_classes, has_classifier=True, **kwargs)


def vgg19_bn_dropout(*, pretrained: bool = False, progress: bool = True, dropout: float = 0.5, conv_dropout: float = 0.0, num_classes: int = 1000, **kwargs: Any) -> VGG:
    """VGG-19-BN model with Dropout.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        dropout (float): Dropout probability for intermediate FC layers and final classifier layer. Default: 0.5.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        num_classes (int): Number of classes in the final classifier layer. Default: 1000. Set to a different value for transfer learning.
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, dropout=dropout, conv_dropout=conv_dropout, num_classes=num_classes, has_classifier=True, **kwargs)


# ----------- FEVGG 系列模型 (Feature Extractor VGG) - 默认无分类器和分类器 Dropout -----------
# 这些模型输出的是 4096 维的特征向量
def fevgg11(*, pretrained: bool = False, progress: bool = True, conv_dropout: float = 0.0, **kwargs: Any) -> VGG:
    """Feature Extractor VGG-11 model (without final classifier layer).
    Outputs a 4096-dimensional feature vector.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    # 强制 dropout=0.0 (因为没有最终分类层，且fevgg系列不应有中间FC的dropout，这里保持一致) 
    # has_classifier=False 表示没有最后的 fc8 层，输出到 fc7 之后 (4096维)
    return _vgg("vgg11", "A", False, pretrained, progress, dropout=0.0, conv_dropout=conv_dropout, num_classes=1000, has_classifier=False, **kwargs)


def fevgg11_bn(*, pretrained: bool = False, progress: bool = True, conv_dropout: float = 0.0, **kwargs: Any) -> VGG:
    """Feature Extractor VGG-11-BN model (without final classifier layer).
    Outputs a 4096-dimensional feature vector.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, dropout=0.0, conv_dropout=conv_dropout, num_classes=1000, has_classifier=False, **kwargs)


def fevgg13(*, pretrained: bool = False, progress: bool = True, conv_dropout: float = 0.0, **kwargs: Any) -> VGG:
    """Feature Extractor VGG-13 model (without final classifier layer).
    Outputs a 4096-dimensional feature vector.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg13", "B", False, pretrained, progress, dropout=0.0, conv_dropout=conv_dropout, num_classes=1000, has_classifier=False, **kwargs)


def fevgg13_bn(*, pretrained: bool = False, progress: bool = True, conv_dropout: float = 0.0, **kwargs: Any) -> VGG:
    """Feature Extractor VGG-13-BN model (without final classifier layer).
    Outputs a 4096-dimensional feature vector.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, dropout=0.0, conv_dropout=conv_dropout, num_classes=1000, has_classifier=False, **kwargs)


def fevgg16(*, pretrained: bool = False, progress: bool = True, conv_dropout: float = 0.0, **kwargs: Any) -> VGG:
    """Feature Extractor VGG-16 model (without final classifier layer).
    Outputs a 4096-dimensional feature vector.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg16", "D", False, pretrained, progress, dropout=0.0, conv_dropout=conv_dropout, num_classes=1000, has_classifier=False, **kwargs)


def fevgg16_bn(*, pretrained: bool = False, progress: bool = True, conv_dropout: float = 0.0, **kwargs: Any) -> VGG:
    """Feature Extractor VGG-16-BN model (without final classifier layer).
    Outputs a 4096-dimensional feature vector.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, dropout=0.0, conv_dropout=conv_dropout, num_classes=1000, has_classifier=False, **kwargs)


def fevgg19(*, pretrained: bool = False, progress: bool = True, conv_dropout: float = 0.0, **kwargs: Any) -> VGG:
    """Feature Extractor VGG-19 model (without final classifier layer).
    Outputs a 4096-dimensional feature vector.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg19", "E", False, pretrained, progress, dropout=0.0, conv_dropout=conv_dropout, num_classes=1000, has_classifier=False, **kwargs)


def fevgg19_bn(*, pretrained: bool = False, progress: bool = True, conv_dropout: float = 0.0, **kwargs: Any) -> VGG:
    """Feature Extractor VGG-19-BN model (without final classifier layer).
    Outputs a 4096-dimensional feature vector.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
        conv_dropout (float): Dropout probability for convolutional layers. Default: 0.0 (no dropout).
        **kwargs: parameters passed to the ``VGG`` base class.
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, dropout=0.0, conv_dropout=conv_dropout, num_classes=1000, has_classifier=False, **kwargs)