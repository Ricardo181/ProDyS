import os
import random
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

class CVC_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip('\n')

        # 构建图像和掩码路径
        img_path = os.path.join(self.data_dir, self.split, 'images', f"{sample_name}.png")
        mask_path = os.path.join(self.data_dir, self.split, 'masks', f"{sample_name}.png")

        # 检查文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图像文件: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"找不到掩码文件: {mask_path}")

        # 使用OpenCV读取灰度图像
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像文件: {img_path}")

        # 将BGR图像转换为RGB图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码（灰度图）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码文件: {mask_path}")

        # 二值化掩码
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        sample = {'image': image, 'label': mask}
        if self.transform:
            sample = self.transform(sample)
            final_image_tensor = sample['image']
            final_label_tensor = sample['label']
        else:
            img_np = sample['image'] # HxWx3 float
            lbl_np = sample['label'] # HxW (0,1) uint8 or other int
            img_np = img_np.astype(np.float32) / 255.0
            final_image_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            final_label_tensor = torch.from_numpy(lbl_np.astype(np.int64))

        if final_label_tensor.dtype != torch.long:
            final_label_tensor = final_label_tensor.long()

        final_sample = {'image': final_image_tensor, 'label': final_label_tensor, 'case_name': sample_name}

        return final_sample