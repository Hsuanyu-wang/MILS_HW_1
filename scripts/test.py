import sys
import os

# 獲取專案根目錄
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

# 導入模型與資料集
# from models.wide_cnn import WideCNN
from models.wide_cnn_new import Residual_WideCNN
from models.resnet_34 import ResNet34
from models.dynamic_conv import ResNet34_Dynamic
from scripts.utils import MiniImageNetDataset

def parse_args():
    parser = argparse.ArgumentParser(description='模型測試腳本')
    parser.add_argument('--model', type=str, default='wide_cnn', choices=['wide_cnn', 'dynamic_conv', 'resnet34'], help='模型類型')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--data_path', type=str, default='/home/MILS_HW1/data/mini_imagenet', help='數據集路徑')
    parser.add_argument('--ckpt', type=str, default=None, help='模型權重路徑（預設自動尋找最佳模型）')
    return parser.parse_args()

def get_model(model_type, in_channels=3, num_classes=100):
    if model_type == 'wide_cnn':
        # return WideCNN(in_channels=in_channels, num_classes=num_classes)
        return Residual_WideCNN(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'dynamic_conv':
        return ResNet34_Dynamic(input_channels=in_channels, num_classes=num_classes)
    elif model_type == 'resnet34':
        return ResNet34(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")

def get_transforms(model_type):
    if model_type == 'resnet34':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_type == 'wide_cnn':
        return transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # dynamic_conv
        return transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = correct / total
    print(f"測試集準確率: {acc:.4f}")
    return acc

def test_dynamic_conv(model, model_path, test_loader, device, channel_combos):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    results = {}
    with torch.no_grad():
        for name, ch_idxs in channel_combos.items():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                # 將不需要的 channel 設為 0
                masked_images = torch.zeros_like(images)
                masked_images[:, ch_idxs, :, :] = images[:, ch_idxs, :, :]
                outputs = model(masked_images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            acc = correct / total
            print(f"[{name}] 測試集準確率: {acc:.4f}")
            results[name] = acc
    return results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(args.model)
    test_path = os.path.join(args.data_path, 'test.txt')
    test_set = MiniImageNetDataset(test_path, transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 載入權重
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join('checkpoints', f'{args.model}_best.pth')
    print(f"載入權重: {ckpt_path}")
            
    if args.model == 'dynamic_conv':
        print(f"初始化{args.model}模型...")
        channel_combos = {
            'R': [0],
            'G': [1],
            'B': [2],
            'RG': [0, 1],
            'GB': [1, 2],
            'RB': [0, 2],
            'RGB': [0, 1, 2]
        }
        model = ResNet34_Dynamic(input_channels=3, num_classes=100).to(device)
        results = test_dynamic_conv(model, ckpt_path, test_loader, device, channel_combos)
        print("results:\n", results)
    else:
        # model = get_model(args.model).to(device)
        # model.load_state_dict(torch.load(ckpt_path, map_location=device))
        # test(model, test_loader, device)
        print(f"初始化{args.model}模型...")
        model = get_model(args.model).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        # 開始測試
        test(model, test_loader, device)

if __name__ == "__main__":
    main()
