
import numpy as np
import torch

class LabelConverter:
    """
    一个用于在数字索引标签和类别名称之间进行转换的类。

    在比较来自不同模型（这些模型可能对同一类别使用不同的索引）的预测时，
    这个类非常有用。通过将所有数字索引标签都转换成标准的类别名称字符串，
    我们可以公平地比较它们的输出。

    如果初始化时没有提供类别名称（class_names is None），
    则转换器会执行“恒等”操作，即直接返回原始输入标签。
    """
    def __init__(self, class_names=None):
        """
        初始化 LabelConverter。

        Args:
            class_names (list or None): 一个字符串列表，其中列表的索引对应于
                                        模型的数字输出标签，值为类别名称。
                                        例如：['猫', '狗', '鸟']
                                        如果为 None，则转换器不执行任何操作。
        """
        if class_names is None:
            # 如果没有提供映射，则设置一个标志，表示不进行任何转换。
            self.mapping = None
        else:
            # 将列表转换为NumPy数组，以便进行高效的索引操作。
            self.mapping = np.array(list(class_names))

    def __call__(self, labels):
        """
        将数字标签转换为类别名称。这使得类的实例可以像函数一样被调用。

        Args:
            labels (torch.Tensor or np.ndarray or int): 
                一个或多个需要转换的数字索引标签。

        Returns:
            np.ndarray or original type: 转换后的类别名称数组。如果未初始化
                                         mapping，则返回原始输入。
        """
        # 如果 self.mapping 为 None，说明不需要转换，直接返回原始标签。
        if self.mapping is None:
            return labels

        # 检查输入是否为PyTorch Tensor，如果是，则将其转移到CPU并转换为NumPy数组。
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # 使用NumPy的高级索引功能，根据输入的数字标签(labels)从self.mapping中查找
        # 对应的类别名称。这对于单个标签和标签数组都有效。
        try:
            return self.mapping[labels]
        except IndexError as e:
            print(f"错误：标签索引 {labels} 超出范围。映射大小为 {len(self.mapping)}。")
            raise e
        except TypeError as e:
            # 如果 labels 是一个无法用于索引的类型（例如浮点数），则会捕获此错误
            print(f"错误：提供的标签 {labels} (类型: {type(labels)}) 不能用于索引。")
            raise e