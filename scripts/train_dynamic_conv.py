import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import argparse

from models.dynamic_conv import ResNet34_Dynamic
from models.resnet_34 import ResNet34
from custom_dataset import TxtImageDataset

def train(args):
    wandb.init(project=args.project, name=args.name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Loader
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    # 在 train() 中替代原先的 ImageFolder 讀法：
    train_set = TxtImageDataset(args.train_txt, args.image_root, transform)
    val_set = TxtImageDataset(args.val_txt, args.image_root, transform)
    # train_set = torchvision.datasets.ImageFolder(root=args.train_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # val_set = torchvision.datasets.ImageFolder(root=args.val_path, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = ResNet34_Dynamic(num_classes=args.num_classes) if args.dynamic else ResNet34(num_classes=args.num_classes)
    model = model.to(device)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        wandb.log({"train_loss": total_loss / len(train_loader), "train_acc": train_acc})

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        wandb.log({"val_loss": val_loss, "val_acc": val_acc})

        print(f"[{epoch+1}/{args.epochs}] Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

    torch.save(model.state_dict(), f"{args.name}_final.pth")

def evaluate(model, loader, criterion, device):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += criterion(out, y).item()
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return loss / len(loader), correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dynamic', action='store_true', help="Use dynamic conv")
    parser.add_argument('--project', type=str, default='visual-signal')
    parser.add_argument('--name', type=str, default='resnet34-experiment')
    parser.add_argument('--image_root', type=str, required=True, help="Path to images/ folder")
    parser.add_argument('--train_txt', type=str, required=True)
    parser.add_argument('--val_txt', type=str, required=True)

    args = parser.parse_args()
    train(args)
