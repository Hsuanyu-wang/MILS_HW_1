import torch
from torch.utils.data import Dataset
from PIL import Image
import os

# class MiniImageNetDataset(Dataset):
#     def __init__(self, txt_file, transform=None):
#         self.samples = []
#         with open(txt_file, 'r') as f:
#             for line in f:
#                 path, label = line.strip().split()
#                 self.samples.append((path, int(label)))
#         self.transform = transform

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, label = self.samples[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label
    
class MiniImageNetDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.samples = []
        self.root_dir = os.path.dirname(txt_file)  # 例如 /home/MILS_HW1/data/mini_imagenet
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(self.root_dir, path)  # 拼接完整路徑
                self.samples.append((full_path, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        assert os.path.exists(img_path), f"Image not found: {img_path}"
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
