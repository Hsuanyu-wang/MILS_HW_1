import sys
import os
# 獲取當前腳本所在目錄的父目錄（專案根目錄）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
import wandb
import argparse

# 根據模型類型導入相應的模型
# from models.wide_cnn import WideCNN
from models.wide_cnn_new import Residual_WideCNN
from models.resnet_34 import ResNet34
from models.dynamic_conv import ResNet34_Dynamic
# from models.dynamic_conv_reg import ResNet34_Dynamic
from scripts.utils import MiniImageNetDataset

def parse_args():
    parser = argparse.ArgumentParser(description='模型訓練腳本')
    parser.add_argument('--model', type=str, default='wide_cnn', choices=['wide_cnn', 'dynamic_conv', 'resnet34'],
                        help='要訓練的模型類型')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='訓練輪數')
    parser.add_argument('--lr', type=float, default=0.001, help='學習率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='優化器類型')
    parser.add_argument('--data_path', type=str, default='/home/MILS_HW1/data/mini_imagenet', help='數據集路徑')
    return parser.parse_args()

def get_model(model_type, num_classes=100):
    """根據模型類型返回相應的模型實例"""
    if model_type == 'wide_cnn':
        # return WideCNN(in_channels=3, num_classes=num_classes)
        return Residual_WideCNN(in_channels=3, num_classes=num_classes)
    elif model_type == 'dynamic_conv':
        return ResNet34_Dynamic(num_classes=num_classes)
    elif model_type == 'resnet34':
        return ResNet34(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")
 
def get_transforms(model_type):
    """根據模型類型返回相應的數據轉換"""
    if model_type == 'resnet34':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_type == 'wide_cnn':  # wide_cnn
        return transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    else:   # dynamic_conv
        return transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
      
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
    """統一的訓練函數"""
    print(f"開始訓練 {args.model} 模型...")
    print(f"訓練設備: {device}")
    print(f"訓練集大小: {len(train_loader.dataset)} 樣本")
    print(f"驗證集大小: {len(val_loader.dataset)} 樣本")
    print(f"批次大小: {train_loader.batch_size}")
    print(f"總訓練批次數: {len(train_loader)}")
    print(f"總驗證批次數: {len(val_loader)}")
    print(f"總訓練輪數: {args.epochs}")

    # 記錄配置到wandb
    wandb.config.update({
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "device": str(device)
    })

    # 確保checkpoints目錄存在
    os.makedirs('checkpoints', exist_ok=True)

    best_acc = 0
    total_start_time = time.time()

    # 添加溫度退火
    initial_temperature = 30
    annealing_epochs = 10
    
    # for epoch in range(args.epochs):
    for epoch in range(args.epochs):
        # 更新溫度
        if epoch < annealing_epochs and hasattr(model, 'update_temperature'):
            model.update_temperature()
            
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} 開始")

        # 訓練階段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        print("訓練階段:")
        train_pbar = tqdm(train_loader, desc=f"訓練 Epoch {epoch+1}/{args.epochs}")
        batch_times = []

        for batch_idx, (images, labels) in enumerate(train_pbar):
            batch_start = time.time()
            # 移動數據到設備
            images, labels = images.to(device), labels.to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            # 計算統計信息
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 計算批次處理時間
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # 更新進度條
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%",
                'time': f"{batch_time:.3f}s"
            })
            
            # 每10個批次或最後一個批次記錄詳細信息
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = train_loss / (batch_idx + 1)
                current_acc = correct / total
                print(f" 批次 {batch_idx+1}/{len(train_loader)} | "
                      f"損失: {avg_loss:.4f} | "
                      f"準確率: {current_acc:.4f} | "
                      f"批次時間: {batch_time:.3f}秒")
                
                # 記錄到wandb
                wandb.log({
                    "epoch": epoch,
                    "train_batch": epoch * len(train_loader) + batch_idx,
                    "train_batch_loss": loss.item(),
                    "train_running_loss": avg_loss,
                    "train_running_acc": current_acc,
                    "train_batch_time": batch_time
                })
        
        # 計算訓練階段統計信息
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total
        train_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        print(f"訓練階段完成 | "
              f"平均損失: {avg_train_loss:.4f} | "
              f"準確率: {train_accuracy:.4f} | "
              f"平均批次時間: {avg_batch_time:.3f}秒 | "
              f"總時間: {train_time:.2f}秒")
        
        # 驗證階段
        val_start_time = time.time()
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        print("\n驗證階段:")
        val_pbar = tqdm(val_loader, desc=f"驗證 Epoch {epoch+1}/{args.epochs}")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_pbar):
                # 移動數據到設備
                images, labels = images.to(device), labels.to(device)
                
                # 前向傳播
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 計算統計信息
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新進度條
                current_acc = correct / total
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*current_acc:.2f}%"
                })
                
                # 每10個批次或最後一個批次記錄詳細信息
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                    print(f" 驗證批次 {batch_idx+1}/{len(val_loader)} | "
                          f"當前準確率: {current_acc:.4f}")
        
        # 計算驗證階段統計信息
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_time = time.time() - val_start_time
        
        print(f"驗證階段完成 | "
              f"平均損失: {avg_val_loss:.4f} | "
              f"準確率: {val_accuracy:.4f} | "
              f"耗時: {val_time:.2f}秒")
        
        # 更新學習率（如果有調度器）
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"學習率從 {old_lr} 調整為 {new_lr}")
        
        # 記錄到wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "train_time": train_time,
            "val_time": val_time,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # 檢查是否為最佳模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            checkpoint_path = os.path.join('checkpoints', f'{args.model}_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"發現更好的模型! 保存到 {checkpoint_path}")
            print(f"新的最佳驗證準確率: {best_acc:.4f}")
            
            # 記錄最佳模型到wandb
            wandb.run.summary["best_val_accuracy"] = best_acc
            wandb.run.summary["best_epoch"] = epoch + 1
        
        # 顯示epoch總結
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} 完成 | "
              f"總耗時: {epoch_time:.2f}秒 | "
              f"當前最佳準確率: {best_acc:.4f}")
    
    # 訓練完成
    total_time = time.time() - total_start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\n{'='*60}")
    print(f"訓練完成! 總耗時: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"最佳驗證準確率: {best_acc:.4f}")
    print(f"最佳模型已保存至 'checkpoints/{args.model}_best.pth'")
    
    # 記錄最終結果到wandb
    wandb.run.summary["total_training_time"] = total_time
    wandb.run.summary["final_val_accuracy"] = val_accuracy
    
    return best_acc

def main():
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化wandb
    wandb.init(project="MILS_HW1", name=f"{args.model}_train")
    
    # 設置數據轉換
    transform = get_transforms(args.model)
    
    # 加載數據集
    print("加載數據集...")
    train_path = os.path.join(args.data_path, 'train.txt')
    val_path = os.path.join(args.data_path, 'val.txt')
    train_set = MiniImageNetDataset(train_path, transform)
    val_set = MiniImageNetDataset(val_path, transform)
    print(f"訓練集: {len(train_set)}個樣本")
    print(f"驗證集: {len(val_set)}個樣本")
    
    # 創建數據加載器
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    # 初始化模型
    print(f"初始化{args.model}模型...")
    model = get_model(args.model).to(device)
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 設置wandb監控模型
    wandb.watch(model, log="all", log_freq=100)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.optimizer == 'sgd':  # sgd
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        print("please choose an appropriate optimizer!\n")
    
    try:
        # 開始訓練
        best_acc = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args)
        print(f"訓練程序完成，最佳準確率: {best_acc:.4f}")
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 確保wandb正常關閉
        wandb.finish()

if __name__ == "__main__":
    main()
