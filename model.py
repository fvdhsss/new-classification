import torch.nn as nn
import torchvision.models as models

def get_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    返回一个基于 ResNet18 的分类模型。
    """
    model = models.resnet18(pretrained=pretrained)
    # 替换最后一层全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
