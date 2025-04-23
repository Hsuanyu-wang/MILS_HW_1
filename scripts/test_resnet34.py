import sys
import os
# 獲取當前腳本所在目錄的父目錄（專案根目錄）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.resnet_34 import ResNet34
from scripts.utils import MiniImageNetDataset
import time
from tqdm import tqdm
import numpy as np

import wandb
wandb.init(project="MILS_HW1", name="resnet34_test_1")

def test():
    print("開始測試過程...")
    
    # 記錄設備信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    wandb.config.update({"device": str(device)})
    
    # 設置數據轉換
    print("配置數據轉換...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    # 加載測試集
    print("加載測試數據集...")
    test_set = MiniImageNetDataset('/home/MILS_HW1/data/mini_imagenet/test.txt', transform)
    print(f"測試集大小: {len(test_set)}個樣本")
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    print(f"測試批次數: {len(test_loader)}")
    
    # 記錄配置到wandb
    wandb.config.update({
        "test_samples": len(test_set),
        "batch_size": 64
    })
    
    # 加載模型
    print("加載預訓練模型...")
    model = ResNet34(num_classes=100).to(device)
    try:
        model.load_state_dict(torch.load('checkpoints/resnet34_best.pth'))
        print("模型加載成功")
    except Exception as e:
        print(f"模型加載失敗: {e}")
        return
    
    # 評估模型
    print("開始評估模型...")
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * 100
    class_total = [0] * 100
    
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    # 使用tqdm顯示進度條
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="測試進度")):
            batch_start = time.time()
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 計算總體準確率
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 計算每個類別的準確率
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (preds[i] == label).item()
                class_total[label] += 1
            
            # 收集預測結果和標籤用於混淆矩陣
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 計算批次處理時間
            batch_time = time.time() - batch_start
            
            # 顯示批次進度
            current_acc = correct / total
            print(f"批次 {batch_idx+1}/{len(test_loader)} | "
                  f"當前準確率: {current_acc:.4f} | "
                  f"處理時間: {batch_time:.4f}秒")
            
            # 記錄到wandb
            wandb.log({
                "batch": batch_idx,
                "batch_accuracy": current_acc,
                "batch_time": batch_time
            })
    
    # 計算總體準確率
    test_accuracy = correct / total
    total_time = time.time() - start_time
    
    # 計算每個類別的準確率
    class_accuracies = {}
    for i in range(100):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            class_accuracies[f"class_{i}_accuracy"] = class_acc
            print(f"類別 {i} 準確率: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    
    # 計算混淆矩陣
    print("計算混淆矩陣...")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 記錄結果到wandb
    print("上傳結果到wandb...")
    wandb.log({
        "test_accuracy": test_accuracy,
        "total_test_time": total_time,
        **class_accuracies
    })
    
    # 創建混淆矩陣圖
    plt_cm = wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=[str(i) for i in range(100)]
    )
    wandb.log({"confusion_matrix": plt_cm})
    
    # 打印最終結果
    print(f"\n{'='*50}")
    print(f"測試完成!")
    print(f"總測試樣本數: {total}")
    print(f"正確預測數: {correct}")
    print(f"測試準確率: {test_accuracy:.4f}")
    print(f"總測試時間: {total_time:.2f}秒")
    
    # 找出表現最好和最差的類別
    best_class = max(range(100), key=lambda i: class_correct[i]/class_total[i] if class_total[i] > 0 else 0)
    worst_class = min(range(100), key=lambda i: class_correct[i]/class_total[i] if class_total[i] > 0 else 1)
    
    print(f"表現最好的類別: {best_class} (準確率: {class_correct[best_class]/class_total[best_class]:.4f})")
    print(f"表現最差的類別: {worst_class} (準確率: {class_correct[worst_class]/class_total[worst_class]:.4f})")
    
    return test_accuracy

if __name__ == "__main__":
    print("開始測試程序")
    try:
        accuracy = test()
        print(f"測試程序完成，最終準確率: {accuracy:.4f}")
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")
    finally:
        # 確保wandb正常關閉
        wandb.finish()
