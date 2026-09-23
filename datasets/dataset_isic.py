import os
import random
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image

class ISIC_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

        # 根据分割类型更新路径
        if self.split == "train":
            self.img_dir = os.path.join(base_dir, 'train', 'images')
            self.mask_dir = os.path.join(base_dir, 'train', 'masks')
        elif self.split == "test":
            self.img_dir = os.path.join(base_dir, 'test', 'images')
            self.mask_dir = os.path.join(base_dir, 'test', 'masks')
        else:
            self.img_dir = os.path.join(base_dir, self.split, 'images')
            self.mask_dir = os.path.join(base_dir, self.split, 'masks')

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_name = self.sample_list[idx].strip('\n')

        # 构建图像和掩码路径 - 文件名已包含ISIC_前缀
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        mask_path_png = os.path.join(self.mask_dir, img_name + '.png')
        mask_path_jpg = os.path.join(self.mask_dir, img_name + '.jpg')
        # 检查文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图像文件: {img_path}")
        if not os.path.exists(mask_path_png) and not os.path.exists(mask_path_jpg):
            raise FileNotFoundError(f"找不到掩码文件: {mask_path}")

        # 加载图像
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像文件: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        mask_path = mask_path_png if os.path.exists(mask_path_png) else mask_path_jpg

        # 加载掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码文件: {mask_path}")

        # 确保掩码是二值化的
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

        final_sample = {'image': final_image_tensor, 'label': final_label_tensor, 'case_name': img_name}
        return final_sample
