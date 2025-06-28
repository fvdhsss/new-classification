import torch

@torch.no_grad()
def accuracy(model, loader, device="cpu"):
    """
    计算模型在数据集上的分类准确率。
    """
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0
