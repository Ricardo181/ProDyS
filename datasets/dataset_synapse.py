import os
import random
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # 修改
        # x, y = image.shape
        x, y, _ = image.shape # Assuming image is HxWxC
        if x != self.output_size[0] or y != self.output_size[1]:
            # 修改
            # image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3) # Keep 3 channels
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # --- 添加归一化 ---
        image = image.astype(np.float32) / 255.0

        # 转换为 Tensor
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0) # 旧代码
        # image = torch.from_numpy(image.astype(np.float32))
        image = torch.from_numpy(image) # Already float32 from normalization
        image = image.permute(2, 0, 1) # HWC -> CHW

        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()} # Ensure label is Long
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name + '.npz')

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到文件: {data_path}")

        data = np.load(data_path)

        image = data['image']  # Numpy array, can be HxW or HxWxC
        mask = data['label']  # Numpy array

        # 二值化掩码
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # 扩展灰度图像为3通道
        image = np.stack([image, image, image], axis=2)
        # Prepare sample dictionary with numpy arrays for the transform
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

        final_sample = {'image': final_image_tensor, 'label': final_label_tensor, 'case_name': slice_name}
        return final_sample
