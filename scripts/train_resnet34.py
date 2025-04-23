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
from models.resnet_34 import ResNet34
from scripts.utils import MiniImageNetDataset

import time
import wandb
wandb.init(project="MILS_HW1", name="resnet34_train_1")

def train():
    print("開始訓練過程...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    print("配置數據轉換...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    print("加載訓練集和驗證集...")
    train_set = MiniImageNetDataset('/home/MILS_HW1/data/mini_imagenet/train.txt', transform)
    val_set = MiniImageNetDataset('/home/MILS_HW1/data/mini_imagenet/val.txt', transform)
    print(f"訓練集大小: {len(train_set)}個樣本")
    print(f"驗證集大小: {len(val_set)}個樣本")
    
    print("創建數據加載器...")
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    print(f"訓練批次數: {len(train_loader)}")
    print(f"驗證批次數: {len(val_loader)}")
    
    print("初始化模型...")
    model = ResNet34(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    print("模型初始化完成")

    print("開始訓練循環...")
    best_acc = 0
    total_start_time = time.time()
    
    for epoch in range(20):
        epoch_start_time = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/20 開始")
        
        # 訓練階段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        print("訓練階段:")
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            images, labels = images.to(device), labels.to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            # 統計
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 批次進度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                batch_time = time.time() - batch_start
                print(f"  批次 {batch_idx+1}/{len(train_loader)} | "
                      f"損失: {loss.item():.4f} | "
                      f"準確率: {100.*correct/total:.2f}% | "
                      f"處理時間: {batch_time:.2f}秒")
        
        # 顯示訓練階段總結
        train_acc = 100. * correct / total
        epoch_train_time = time.time() - epoch_start_time
        print(f"訓練階段完成 | 平均損失: {train_loss/len(train_loader):.4f} | "
              f"準確率: {train_acc:.2f}% | 耗時: {epoch_train_time:.2f}秒")
        
        # 驗證階段
        val_start_time = time.time()
        print("\n驗證階段:")
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 驗證批次進度
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                    print(f"  驗證批次 {batch_idx+1}/{len(val_loader)} 處理完成")
        
        # 計算驗證準確率
        val_acc = 100. * correct / total
        val_time = time.time() - val_start_time
        print(f"驗證階段完成 | 平均損失: {val_loss/len(val_loader):.4f} | "
              f"準確率: {val_acc:.2f}% | 耗時: {val_time:.2f}秒")
        
        # 檢查是否為最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            print("發現更好的模型! 保存檢查點...")
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/resnet34_best.pth')
            print("模型保存完成")
        
        # 更新學習率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"學習率從 {old_lr} 調整為 {new_lr}")
        
        # 顯示epoch總結
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} 完成 | 總耗時: {epoch_time:.2f}秒")
        print(f"當前最佳驗證準確率: {best_acc:.2f}%")
        
        wandb.log({"train_loss": (val_loss/len(val_loader)), "accuracy": val_acc})
    
    # 訓練完成
    total_time = time.time() - total_start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n{'='*50}")
    print(f"訓練完成! 總耗時: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"最佳驗證準確率: {best_acc:.2f}%")
    print(f"最佳模型已保存至 'checkpoints/resnet34_best.pth'")

if __name__ == "__main__":
    print("開始訓練程序")
    train()
