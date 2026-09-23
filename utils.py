import os
import torch.nn.functional as F
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import copy
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
from scipy.spatial import cKDTree

def dice_coefficient(a, b):
    """计算两个numpy数组之间的Dice系数"""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)

    # 计算交集
    intersection = np.logical_and(a, b)
    if a.sum() + b.sum() != 0.0:
        return 2. * intersection.sum() / (a.sum() + b.sum())
    else:
        return 1.0

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    if pred.shape != gt.shape:
        raise ValueError("Shape mismatch: pred and gt must have the same shape.")

    # 计算 TP, FP, TN, FN
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    # 计算指标
    # 避免除以零
    epsilon = 1e-6

    se = tp / (tp + fn + epsilon) # 灵敏度 (Sensitivity) / Recall
    sp = tn / (tn + fp + epsilon) # 特异度 (Specificity)
    acc = (tp + tn) / (tp + tn + fp + fn + epsilon) # 准确率 (Accuracy)
    iou = tp / (tp + fp + fn + epsilon) # 交并比 (Intersection over Union)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon) # Dice 系数 (与 dice_coefficient 函数结果相同)

    # HD95 计算需要额外的库 (medpy)，仅在两者都存在前景时计算
    if pred.sum() > 0 and gt.sum() > 0:
        # 确保输入是二进制的 (medpy 要求)
        pred_binary = pred.astype(np.uint8)
        gt_binary = gt.astype(np.uint8)
        # try-except 块用于处理 HD 计算可能出现的异常
        try:
            hd95 = metric.binary.hd95(pred_binary, gt_binary)
        except RuntimeError:
            # 例如，如果其中一个掩码完全为空或完全填满，可能会引发错误
            hd95 = np.nan # 或者其他你认为合适的错误值
    elif pred.sum() == 0 and gt.sum() == 0:
        # 如果两者都为空，则HD95未定义，或者可以认为是0
        hd95 = 0.0
    else:
        # 如果一个为空另一个不为空，HD95可能无意义或非常大
        # 在某些情况下，将其设为图像对角线长度或一个大值
        # 这里我们返回 np.nan 表示无法有效计算
        hd95 = np.nan

    # 返回所有指标
    # 如果 gt 全为背景 (fn=0, tp=0)，则 se=0。如果gt全为前景(tn=0, fp=0)，则sp=0。
    # 如果 pred 和 gt 都全为背景 (tp=0, fn=0, fp=0)，则 se=0, iou=0, dice=0。sp=1, acc=1
    # 如果 pred 和 gt 都全为前景 (tn=0, fp=0, fn=0)，则 sp=0。se=1, iou=1, dice=1, acc=1
    # 如果 pred 全为背景 (tp=0, fp=0)，则 iou=0, dice=0。se=0。
    # 如果 pred 全为前景 (tn=0, fn=0)，则 sp=0。

    # 处理特殊情况返回值 (确保逻辑一致性)
    if gt.sum() == 0: # Ground Truth 全是背景
        if pred.sum() == 0: # 预测也全是背景
            se = 1.0 # 技术上未定义，但常视为1
            sp = 1.0
            acc = 1.0
            iou = 1.0 # 或者 1.0，取决于定义
            dice = 1.0
            hd95 = 0.0
        else: # 预测有前景 (全是 FP)
            se = 0.0 # 技术上未定义，但常视为0
            sp = 0.0
            acc = tn / (tn + fp + epsilon) # tn / total
            iou = 0.0
            dice = 0.0
            hd95 = np.nan # 或者一个大值

    return dice, hd95, se, sp, acc, iou

# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, cuda=0):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

#     # 获取原始图像的维度 (通常是 H, W，或者 C, H, W 中的 H, W)
#     # 假设 image 是 (num_slices_or_channels, H, W) 结构，经过 squeeze 后
#     if image.ndim == 3: # 例如 (C, H, W)
#         original_height, original_width = image.shape[1], image.shape[2]
#         # 如果只有一个通道，则去除通道维度，使其变为 (H,W) 以便与 patch_size[0], patch_size[1] 比较
#         if image.shape[0] == 1:
#             image_for_network_input = image.squeeze(0)
#         else:
#             # 如果多通道，zoom 时需要考虑通道，但通常分割网络处理单通道或特定通道组合
#             # 此处假设网络输入是基于 H, W 的，多通道图像的缩放因子应用在 H, W 维度
#             image_for_network_input = image
#     elif image.ndim == 2: # (H, W)
#         original_height, original_width = image.shape[0], image.shape[1]
#         image_for_network_input = image
#     else:
#         raise ValueError(f"Unsupported image dimensions: {image.ndim}")

#     # 1. 如果原始图像尺寸与网络期望的 patch_size 不同，则缩放图像以匹配 patch_size
#     if original_height != patch_size[0] or original_width != patch_size[1]:
#         if image_for_network_input.ndim == 3: # (C, H, W)
#             zoom_factors_img = (1, patch_size[0] / original_height, patch_size[1] / original_width)
#         else: # (H, W)
#             zoom_factors_img = (patch_size[0] / original_height, patch_size[1] / original_width)
#         image_resized_for_network = zoom(image_for_network_input, zoom_factors_img, order=3)
#     else:
#         image_resized_for_network = image_for_network_input.copy() # 使用副本

#     input_tensor = torch.from_numpy(image_resized_for_network).unsqueeze(0).float().cuda(cuda)
#     if input_tensor.ndim == 3: # 如果image_resized_for_network是(H,W), unsqueeze(0)后是(1,H,W), 网络可能要(1,C,H,W)
#         input_tensor = input_tensor.unsqueeze(1) # 变为 (1,1,H,W)
#     elif input_tensor.ndim == 4 and input_tensor.shape[1] != 1 and image.ndim == 3 and image.shape[0]!=1 : # (1,C,H,W) but C from original multi-channel image
#         pass # 已经是 (1,C,H,W)
#     elif input_tensor.ndim == 4 and image.ndim == 2 : # image was (H,W) -> (1,H,W) -> (1,1,H,W)
#          pass


#     net.eval()
#     with torch.no_grad():
#         out_prob_network = torch.softmax(net(input_tensor), dim=1) # Shape: (1, num_classes, patch_h, patch_w)

#         # 2. 提取前景概率图 (在 patch_size 维度)
#         # prob_map_at_patch_size 的形状是 (patch_h, patch_w)，例如 224x224
#         prob_map_at_patch_size = out_prob_network[:, 1, :, :].squeeze(0).cpu().detach().numpy()

#         # --- 修改开始: 在 patch_size 分辨率下进行评估 ---

#         # 3. 对 patch_size 的概率图进行二值化 (使用0.5阈值)
#         # prediction_binary_at_eval_res 的形状是 (patch_h, patch_w)
#         prediction_binary_at_eval_res = (prob_map_at_patch_size > 0.5).astype(np.uint8)

#         # 4. 准备真实标签 (label)，并将其缩放到 patch_size (eval_res)
#         # label 是函数开始时获取的原始高分辨率标签 (例如 512x512)
#         # patch_size[0] 是评估目标高度 (例如 224), patch_size[1] 是评估目标宽度 (例如 224)

#         gt_label_for_eval = label # 使用函数开始时传入的原始标签
#         if gt_label_for_eval.ndim == 3: # 处理可能的 (C, H, W) 格式
#             if gt_label_for_eval.shape[0] == 1:
#                 gt_label_for_eval = gt_label_for_eval.squeeze(0)
#             else:
#                 # 如果标签有多个通道，这里默认取第一个通道
#                 # 你可能需要根据具体情况调整
#                 gt_label_for_eval = gt_label_for_eval[0]

#         if gt_label_for_eval.ndim != 2:
#             raise ValueError(f"Ground truth label for evaluation is not 2D: shape {gt_label_for_eval.shape}")

#         # 将真实标签缩放到 patch_size (评估分辨率)
#         if gt_label_for_eval.shape[0] != patch_size[0] or gt_label_for_eval.shape[1] != patch_size[1]:
#             label_resized_to_eval_res = zoom(gt_label_for_eval,
#                                              (patch_size[0] / gt_label_for_eval.shape[0],
#                                               patch_size[1] / gt_label_for_eval.shape[1]),
#                                              order=0) # 最近邻插值用于掩码
#         else:
#             label_resized_to_eval_res = gt_label_for_eval.copy()

#         # 确保缩放后的标签是二值的 (0/1)
#         label_resized_to_eval_res = (label_resized_to_eval_res > 0.5).astype(np.uint8)

#         # --- 修改结束: 在 patch_size 分辨率下进行评估 ---

#         # 为了保存和可视化，我们仍然可以将预测上采样回原始尺寸
#         # (这部分不影响指标计算，指标计算使用 eval_res 的版本)
#         if prob_map_at_patch_size.shape[0] != original_height or prob_map_at_patch_size.shape[1] != original_width:
#             prob_map_resized_for_saving = zoom(prob_map_at_patch_size, # 使用原始概率图进行缩放以获得更平滑的可视化
#                                                 (original_height / prob_map_at_patch_size.shape[0],
#                                                  original_width / prob_map_at_patch_size.shape[1]),
#                                                 order=3)
#         else:
#             prob_map_resized_for_saving = prob_map_at_patch_size

#         prediction_binary_for_saving = (prob_map_resized_for_saving > 0.5).astype(np.uint8)

#         # 准备原始尺寸的真实标签用于保存
#         original_gt_label_for_saving = label # 使用函数开始时的原始标签
#         if original_gt_label_for_saving.ndim == 3:
#             if original_gt_label_for_saving.shape[0] == 1:
#                 original_gt_label_for_saving = original_gt_label_for_saving.squeeze(0)
#             else:
#                 original_gt_label_for_saving = original_gt_label_for_saving[0]

#         if original_gt_label_for_saving.shape[0] != original_height or original_gt_label_for_saving.shape[1] != original_width:
#              # 如果原始标签尺寸与图像不一致 (不太可能在此处发生，但作为健壮性检查)
#             original_gt_label_for_saving = zoom(original_gt_label_for_saving,
#                                  (original_height / original_gt_label_for_saving.shape[0],
#                                   original_width / original_gt_label_for_saving.shape[1]),
#                                  order=0)
#         original_gt_label_for_saving = (original_gt_label_for_saving > 0.5).astype(np.uint8)


#         # 创建结果目录 (如果尚不存在)
#         if not os.path.exists('pred_image'):
#             os.makedirs('pred_image')
#         if not os.path.exists('label_image'):
#             os.makedirs('label_image')

#         # 保存图像 (使用上采样到原始尺寸的预测和原始尺寸的标签进行可视化)
#         filename, extension = os.path.splitext(case)
#         plt.imsave(os.path.join('pred_image', filename) + '.png', prediction_binary_for_saving, cmap='gray', vmin=0, vmax=1)
#         plt.imsave(os.path.join('label_image', filename) + '.png', original_gt_label_for_saving, cmap='gray', vmin=0, vmax=1)

#         # 打印单个Dice (可选，现在基于评估分辨率)
#         # print('dice @ eval_res:' + str(dice_coefficient(prediction_binary_at_eval_res, label_resized_to_eval_res)))

#     metric_list = []
#     for i in range(1, classes): # 假设类别0是背景
#         # 使用在评估分辨率 (patch_size) 下的二值图计算指标
#         metric_list.append(calculate_metric_percase(prediction_binary_at_eval_res == i, label_resized_to_eval_res == i))

#     return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, cuda=0):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # 获取原始图像尺寸
    if image.ndim == 3:
        original_height, original_width = image.shape[1], image.shape[2]
        if image.shape[0] == 1:
            image_for_network_input = image.squeeze(0)
        else:
            image_for_network_input = image
    elif image.ndim == 2:
        original_height, original_width = image.shape[0], image.shape[1]
        image_for_network_input = image
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")

    # 缩放到 patch_size
    if original_height != patch_size[0] or original_width != patch_size[1]:
        if image_for_network_input.ndim == 3:
            zoom_factors_img = (1, patch_size[0] / original_height, patch_size[1] / original_width)
        else:
            zoom_factors_img = (patch_size[0] / original_height, patch_size[1] / original_width)
        image_resized_for_network = zoom(image_for_network_input, zoom_factors_img, order=3)
    else:
        image_resized_for_network = image_for_network_input.copy()

    input_tensor = torch.from_numpy(image_resized_for_network).unsqueeze(0).float().cuda(cuda)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(1)

    net.eval()
    with torch.no_grad():
        out_prob_network = torch.softmax(net(input_tensor), dim=1)
        prob_map_at_patch_size = out_prob_network[:, 1, :, :].squeeze(0).cpu().detach().numpy()
        prediction_binary_at_eval_res = (prob_map_at_patch_size > 0.5).astype(np.uint8)

        gt_label_for_eval = label
        if gt_label_for_eval.ndim == 3:
            if gt_label_for_eval.shape[0] == 1:
                gt_label_for_eval = gt_label_for_eval.squeeze(0)
            else:
                gt_label_for_eval = gt_label_for_eval[0]
        if gt_label_for_eval.ndim != 2:
            raise ValueError(f"Ground truth label for evaluation is not 2D: shape {gt_label_for_eval.shape}")

        if gt_label_for_eval.shape[0] != patch_size[0] or gt_label_for_eval.shape[1] != patch_size[1]:
            label_resized_to_eval_res = zoom(gt_label_for_eval,
                                             (patch_size[0] / gt_label_for_eval.shape[0],
                                              patch_size[1] / gt_label_for_eval.shape[1]),
                                             order=0)
        else:
            label_resized_to_eval_res = gt_label_for_eval.copy()
        label_resized_to_eval_res = (label_resized_to_eval_res > 0.5).astype(np.uint8)

        # 上采样回原始尺寸用于保存
        if prob_map_at_patch_size.shape[0] != original_height or prob_map_at_patch_size.shape[1] != original_width:
            prob_map_resized_for_saving = zoom(prob_map_at_patch_size,
                                               (original_height / prob_map_at_patch_size.shape[0],
                                                original_width / prob_map_at_patch_size.shape[1]),
                                               order=3)
        else:
            prob_map_resized_for_saving = prob_map_at_patch_size
        prediction_binary_for_saving = (prob_map_resized_for_saving > 0.5).astype(np.uint8)

        original_gt_label_for_saving = label
        if original_gt_label_for_saving.ndim == 3:
            if original_gt_label_for_saving.shape[0] == 1:
                original_gt_label_for_saving = original_gt_label_for_saving.squeeze(0)
            else:
                original_gt_label_for_saving = original_gt_label_for_saving[0]
        if original_gt_label_for_saving.shape[0] != original_height or original_gt_label_for_saving.shape[1] != original_width:
            original_gt_label_for_saving = zoom(original_gt_label_for_saving,
                                                (original_height / original_gt_label_for_saving.shape[0],
                                                 original_width / original_gt_label_for_saving.shape[1]),
                                                order=0)
        original_gt_label_for_saving = (original_gt_label_for_saving > 0.5).astype(np.uint8)

        # --- 创建目录 ---
        if not os.path.exists('pred_image'):
            os.makedirs('pred_image')
        if not os.path.exists('label_image'):
            os.makedirs('label_image')
        if not os.path.exists('original'):
            os.makedirs('original')  # 新增：保存CT原图

        filename, extension = os.path.splitext(case)

        # --- 保存CT原图 ---
        if image.ndim == 3:
            ct_image_to_save = np.transpose(image_for_network_input, (1, 2, 0))
        else:
            ct_image_to_save = image_for_network_input
        ct_image_norm = (ct_image_to_save - ct_image_to_save.min()) / (ct_image_to_save.max() - ct_image_to_save.min())
        plt.imsave(os.path.join('original', filename) + '.png', ct_image_norm, cmap='gray', vmin=0, vmax=1)

        # --- 保存预测掩码和标签 ---
        plt.imsave(os.path.join('pred_image', filename) + '.png', prediction_binary_for_saving, cmap='gray', vmin=0, vmax=1)
        plt.imsave(os.path.join('label_image', filename) + '.png', original_gt_label_for_saving, cmap='gray', vmin=0, vmax=1)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction_binary_at_eval_res == i, label_resized_to_eval_res == i))

    return metric_list
