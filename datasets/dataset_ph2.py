import os
import random
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

class PH2_dataset(Dataset):
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
        img_path = os.path.join(self.data_dir, self.split, 'images', f"{sample_name}.bmp")
        mask_path = os.path.join(self.data_dir, self.split, 'masks', f"{sample_name}_lesion.bmp")

        # 检查文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图像文件: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"找不到掩码文件: {mask_path}")

        # 使用OpenCV读取彩色图像 (BGR格式)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像文件: {img_path}")

        # 将BGR图像转换为RGB图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码（灰度图）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码文件: {mask_path}")

        # 二值化掩码 (输出 numpy 数组, dtype 可能是 uint8, 值为 0 或 1)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # 初始样本（numpy 数组）
        current_sample_np = {'image': image, 'label': mask}

        if self.transform:
            # transform (RandomGenerator) 应该接收 numpy 数组并输出 tensor
            # RandomGenerator 内部会将 label 转为 .long()
            transformed_sample = self.transform(current_sample_np)
            final_image_tensor = transformed_sample['image']
            final_label_tensor = transformed_sample['label']
        else:
            # 如果没有 transform (通常是 test/validation)
            # 将 numpy 数组转换为 tensor，并确保类型正确
            img_np = current_sample_np['image'] # HxWx3 float
            lbl_np = current_sample_np['label'] # HxW (0,1) uint8 or other int

            img_np = img_np.astype(np.float32) / 255.0

            final_image_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # CHW float32
            final_label_tensor = torch.from_numpy(lbl_np.astype(np.int64)) # HW int64 (LongTensor)

        # 确保最终的标签是 LongTensor
        # RandomGenerator 已经处理了，这里主要是为了 else 分支以及双重检查
        if final_label_tensor.dtype != torch.long:
            final_label_tensor = final_label_tensor.long()

        final_sample = {'image': final_image_tensor, 'label': final_label_tensor, 'case_name': sample_name}

        return final_sample
