import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
import cv2

class Thyroid_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

        # 针对甲状腺数据集的特殊文件夹结构
        if split == "train":
            self.img_dir = os.path.join(base_dir, "train","images")
            self.label_dir = os.path.join(base_dir, "train","masks")
        else:
            self.img_dir = os.path.join(base_dir, "test","images")
            self.label_dir = os.path.join(base_dir, "test","masks")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_name = self.sample_list[idx].strip('\n')

        # 构建图像和掩码路径
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        label_path = os.path.join(self.label_dir, img_name + '.jpg')

        # 检查文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图像文件: {img_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"找不到掩码文件: {label_path}")

        # 读取图像和标签
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图像文件: {img_path}")

        # 读取标签（假设标签是灰度图像）
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"无法读取掩码文件: {label_path}")

        # 确保标签是二值化的
        _, label = cv2.threshold(label, 127, 1, cv2.THRESH_BINARY)

        # 扩展灰度图像为3通道
        image = np.stack([image, image, image], axis=2)

        sample = {'image': image, 'label': label }

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

        final_sample = {'image': final_image_tensor, 'label': final_label_tensor, 'case_name': img_name}
        return final_sample
