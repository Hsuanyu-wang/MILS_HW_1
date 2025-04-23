# Assignment1: Visual Signal Processing

## 專案簡介
本專案為 mini-ImageNet 圖像分類作業，包含：
- Task A：可變輸入通道的動態卷積模組設計與實驗
    
    [Dynamic Convolution: Attention over Convolution Kernels](https://arxiv.org/abs/1912.03458)
- Task B：2~4 有效層的影像分類網路設計與驗證


## 資料集
- mini-ImageNet（[下載連結](https://cchsu.info/files/images.zip)）
- `data/` 資料夾下有 train.txt, validation.txt, test.txt

## 安裝與執行
```
git clone https://github.com/Hsuanyu-wang/MILS_HW_1.git
pip install -r requirements.txt
python scripts/train.py
python scripts/test.py
```
model:['resnet34', 'dynamic_conv', ''wide_cnn]
optimizer:[]

train.py usage:
```
# 訓練 WideCNN 模型
python train.py --model wide_cnn --optimizer adam --lr 0.001

# 訓練 Dynamic Conv 模型
python train.py --model dynamic_conv --optimizer adam --lr 0.001

# 訓練 ResNet34 模型
python train.py --model resnet34 --optimizer adam --lr 0.001
```

test.py usage:
```
# 測試 WideCNN 模型
python test.py --model wide_cnn --ckpt /home/MILS_HW1/scripts/checkpoints/wide_cnn_best.pth

# 測試 Dynamic Conv 模型
python test.py --model dynamic_conv --ckpt /home/MILS_HW1/scripts/checkpoints/dynamic_conv_best.pth

# 測試 ResNet34 模型
python test.py --model resnet34 --ckpt /home/MILS_HW1/scripts/checkpoints/resnet34_best.pth
```