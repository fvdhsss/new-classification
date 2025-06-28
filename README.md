# Simple Image Classification
一个基于 PyTorch + ResNet18 的简单图像分类示例项目。

## 特性 feature

- 使用 `torchvision.datasets.ImageFolder` 组织数据，目录结构如下：

- 基于 ResNet18，替换最后一层全连接以适应任意类别数。
- 默认使用 Adam 优化器和交叉熵损失函数。
- 支持 GPU / CPU 自动切换。

## 安装 install

```bash
git clone https://github.com/fvdhsss/image-classification.git
cd image-classification
pip install -r requirements.txt
```
## a quick atart
```bash
python train.py \ --data-dir data \ --num-classes 2 \  --epochs 20 \ --batch-size 64 \ --lr 0.001
