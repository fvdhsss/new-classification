import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model
from utils import accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Simple Image Classification")
    parser.add_argument("--data-dir", type=str, required=True, help="训练/验证数据所在目录，子文件夹为类别名")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(f"{args.data_dir}/train", transform=transform)
    val_ds   = datasets.ImageFolder(f"{args.data_dir}/val",   transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # 模型、损失、优化器
    model = get_model(args.num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        # 训练
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        val_acc = accuracy(model, val_loader, device=args.device)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"Train Loss: {total_loss/len(train_loader):.4f}  "
              f"Val Acc: {val_acc*100:.2f}%")

if __name__ == "__main__":
    main()
