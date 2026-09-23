import math
import cv2
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import init
import torchvision.ops as ops
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Union  # 用于类型提示

from threading import Lock
from contextlib import contextmanager, nullcontext  # 确保nullcontext可用
import faiss
# 创建专属于swin_transformer模块的logger
swin_logger = logging.getLogger('swin_transformer')
swin_logger.setLevel(logging.INFO)

# 清除已有的处理器，避免重复添加
for handler in swin_logger.handlers[:]:
    swin_logger.removeHandler(handler)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
swin_logger.addHandler(console_handler)

# 禁用根日志处理器的传播，避免日志重复
swin_logger.propagate = False

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            # 创建网格索引以代替嵌套循环
            h_indices = torch.zeros(len(h_slices), dtype=torch.long)
            w_indices = torch.zeros(len(w_slices), dtype=torch.long)

            # 使用网格索引替代嵌套循环，一次性计算掩码值
            cnt = 0
            # 创建网格坐标
            h_idx, w_idx = torch.meshgrid(torch.arange(len(h_slices)), torch.arange(len(w_slices)), indexing='ij')
            h_idx = h_idx.flatten()
            w_idx = w_idx.flatten()

            # 为每个区域分配掩码值
            for i in range(len(h_idx)):
                img_mask[:, h_slices[h_idx[i]], w_slices[w_idx[i]], :] = i

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)


    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x

def expend_as(tensor, rep):
    return tensor.repeat(1, rep, 1, 1)

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        super(SpatialBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=size, padding=(size // 2)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x, channel_data):
        device = x.device
        conv1 = self.conv1(x)
        spatial_data = self.conv2(conv1)

        data3 = torch.add(channel_data, spatial_data)
        data3 = torch.relu(data3)
        # 使用 nn.Conv2d 创建卷积层，并确保它在正确的设备上
        conv_layer = nn.Conv2d(data3.size(1), 1, kernel_size=1, padding=0).cuda(device)
        data3 = torch.sigmoid(conv_layer(data3))

        a = expend_as(data3, channel_data.size(1))
        y = a * channel_data

        a1 = expend_as(1 - data3, spatial_data.size(1))
        y1 = a1 * spatial_data

        combined = torch.cat([y, y1], dim=1)
        out = self.final_conv(combined)

        return out

class LaplacianConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(LaplacianConvLayer, self).__init__()
        # 定义拉普拉斯算子核 (3x3)
        laplacian_kernel = torch.tensor([[0., 1., 0.],
                                         [1., -8., 1.],
                                         [0., 1., 0.]], dtype=torch.float32)

        # 调整为四维张量 [1, 1, 3, 3]（单通道，3x3核）
        laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        # 根据输入和输出通道数，创建卷积核
        # [out_channels, in_channels, kernel_height, kernel_width]
        self.laplacian_kernel = nn.Parameter(
            laplacian_kernel.repeat(out_channels, in_channels, 1, 1), requires_grad=False
        )

    def forward(self, x):
        # 使用 F.conv2d 进行卷积操作，使用拉普拉斯算子核
        return F.conv2d(x, self.laplacian_kernel, padding=1)

# 主模块：包含三个并行分支和空间注意力
class ParallelBranchesBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18],kernel_size=[3,5,7],map_reduce=8,stride=1):
        super(ParallelBranchesBlock, self).__init__()
        self.out_channels = out_channels # 输出通道数
        # 每个分支的卷积块
        self.branches = nn.ModuleList()
        for i in range(3):
            branch = nn.Sequential(
                BasicConv(in_channels= in_channels, out_channels= out_channels, kernel_size= 1,padding= 0,stride=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilations[i], padding=dilations[i]),
                ECAAttention(kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size[i],padding= i+1),
                LaplacianConvLayer(out_channels,out_channels),
                BasicConv(in_channels= in_channels, out_channels= out_channels, kernel_size= 1,padding= 0,stride=1),
            )
            self.branches.append(branch)

        # 空间注意力模块
        self.spatial_attention = SpatialBlock(out_channels, out_channels)
        # 在每个分支的输出后加一个卷积层
        self.extra_conv =BasicConv(in_channels= out_channels, out_channels= out_channels, kernel_size= 1,padding= 0,stride=1)
        self.concat_conv =nn.Conv2d(out_channels * 3,out_channels, kernel_size= 1,padding= 0,stride=1)

    def forward(self, x):
        x_shortcut = x

        # 对每个分支分别进行前向传播
        branch_outputs = [branch(x) for branch in self.branches]
        # 对每个分支的输出计算通道注意力

        concatenated_output = torch.cat(branch_outputs, dim=1)
        out2 = self.concat_conv(concatenated_output)
        # 将三个分支的输出进行相加
        out = sum(branch_outputs)
        out3 = out + x_shortcut + out2
        combined = self.extra_conv(out3)
        # 经过空间注意力模块
        combined = self.spatial_attention(combined, x)

        return combined

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x_shortcut = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        y = x_shortcut + x # Residual connection
        return y

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, flag=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            # 修复：检查downsample是否是一个类或实例
            if isinstance(downsample, type):
                # 如果是类，创建它的实例
                self.downsample = downsample(input_resolution=input_resolution, dim=dim, norm_layer=norm_layer)
            else:
                # 如果已经是实例，直接使用
                self.downsample = downsample
        else:
            self.downsample = None

        self.flag = flag
        self.convBottleneckBlock = ConvBottleneckBlock(768, 768, 768)

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            # else:
            #     x = blk(x)
            else:
                if not self.flag:
                    x = blk(x)
        if self.flag:
            # print(x.shape)
            x = x.reshape(-1, 7, 7, 768)  # iteration 7:([10, 768, 7, 7])
            # print(x.shape)
            x = x.permute(0, 3, 1, 2)
            # print(x.shape)
            x = self.convBottleneckBlock(x)
            # print(x.shape)
        if self.downsample is not None:
            x = self.downsample(x)
            # print(x.shape)
        return x


class ConvBottleneckBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class PatchMerging(nn.Module):
    r""" 标准的Patch合并层，用于下采样特征图。

    当use_feature_bank=False时使用此类作为下采样层。

    与PatchMergingWithFeatureBank相比:
    - 更简单: 只执行基本的特征合并和线性投影
    - 更快速: 没有额外的特征银行逻辑，计算开销更小
    - 不保留历史特征: 每次只处理当前输入的特征

    Args:
        input_resolution (tuple[int]): 输入特征图的分辨率.
        dim (int): 输入通道数.
        norm_layer (nn.Module, optional): 规范化层. 默认: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

# 定义ECA注意力模块的类
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 对Conv2d层使用Kaiming初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 批归一化层权重初始化为1
                init.constant_(m.bias, 0)  # 批归一化层偏置初始化为0
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 全连接层权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 全连接层偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出
        y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        return x * y.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法

class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first",
                 use_feature_bank=True, use_enhanced_up_x4=True, use_pbb=True, # 功能开关参数
                 **kwargs):
        super().__init__()

        # 存储功能开关参数
        self.use_feature_bank = use_feature_bank
        self.use_enhanced_up_x4 = use_enhanced_up_x4
        self.use_pbb = use_pbb

        # 记录基本参数
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patches_resolution = None  # 将在patch_embed初始化后设置

        # 初始化特征增强模块
        self._init_feature_enhancement(embed_dim)

        # 记录初始化信息
        swin_logger.info(
            "SwinTransformerSys initializing with settings:\n"
            f"  depths={depths}, depths_decoder={depths_decoder}\n"
            f"  drop_path_rate={drop_path_rate}, num_classes={num_classes}\n"
            f"  use_feature_bank={self.use_feature_bank}\n"
            f"  use_enhanced_up_x4={self.use_enhanced_up_x4}\n"
            f"  use_pbb={self.use_pbb}"
        )

        # 分割图像为非重叠的patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置嵌入
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 创建下采样层（根据use_feature_bank选择不同的实现）
        self.feature_bank_layers = self._create_downsample_layers(
            patches_resolution, embed_dim, norm_layer)

        # 构建编码器和瓶颈层
        self.layers = self._create_encoder_layers(
            patches_resolution, embed_dim, depths, num_heads, window_size,
            dpr, norm_layer, drop_rate, attn_drop_rate, qkv_bias, qk_scale, use_checkpoint)

        # 创建并行分支模块（如果use_pbb=True）
        self.pbb_list = self._create_pbb_modules(embed_dim) if self.use_pbb else None

        # 构建解码器层
        self.layers_up, self.concat_back_dim = self._create_decoder_layers(
            patches_resolution, embed_dim, depths, depths_decoder, num_heads, window_size,
            dpr, norm_layer, drop_rate, attn_drop_rate, qkv_bias, qk_scale, use_checkpoint)

        # 规范化层
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        # 根据use_enhanced_up_x4初始化上采样模块
        self._init_upsampling_modules(img_size, patch_size, embed_dim, norm_layer)

        # 输出层（两种上采样方式都需要）
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        # 初始化权重
        self.apply(self._init_weights)

        # UNet++ 跳跃连接模块
        self._init_unetpp_modules(embed_dim)

        # 解码列表（用于存储中间特征）
        self.decode_list = []

    def _init_feature_enhancement(self, embed_dim):
        """初始化特征增强相关的模块和参数"""
        # 特征增强模块
        self.enhance_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.enhance_bn = nn.BatchNorm2d(embed_dim)
        self.fusion_weight = nn.Linear(embed_dim, 1)

        # 动态参数
        self.attention_channel_weights = nn.Parameter(torch.ones(3) * 0.5)
        self.attention_spatial_weights = nn.Parameter(torch.ones(3) * 0.5)
        self.attention_edge_weights = nn.Parameter(torch.ones(3) * 0.3)
        self.fusion_threshold = nn.Parameter(torch.tensor(0.35))
        self.layer_decay_base = nn.Parameter(torch.tensor(0.7))
        self.layer_importance_base = nn.Parameter(torch.tensor(0.2))
        self.layer_importance_scale = nn.Parameter(torch.tensor(0.25))
        self.consistency_kernel_size = nn.Parameter(torch.tensor(3.0))
        self.global_context_scale = nn.Parameter(torch.tensor(0.1))
        self.enhanced_feature_scale = nn.Parameter(torch.tensor(0.5))

        # Sobel算子核作为缓冲区
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x_kernel, persistent=False)
        self.register_buffer('sobel_y', sobel_y_kernel, persistent=False)

        # 初始化权重
        nn.init.kaiming_normal_(self.enhance_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.enhance_bn.weight, 1)
        nn.init.constant_(self.enhance_bn.bias, 0)
        nn.init.xavier_normal_(self.fusion_weight.weight)
        nn.init.constant_(self.fusion_weight.bias, 0)

    def _create_downsample_layers(self, patches_resolution, embed_dim, norm_layer):
        """创建下采样层，根据use_feature_bank选择不同的实现"""
        feature_bank_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            if i_layer < self.num_layers - 1:  # 最后一层不需要下采样
                resolution = (patches_resolution[0] // (2 ** i_layer),
                             patches_resolution[1] // (2 ** i_layer))
                dim = int(embed_dim * 2 ** i_layer)

                if self.use_feature_bank:
                    # 使用特征银行增强的下采样层
                    merging_layer = PatchMergingWithFeatureBank(
                        input_resolution=resolution,
                        dim=dim,
                        norm_layer=norm_layer,
                    )
                    swin_logger.info(f"层 {i_layer}: 使用特征银行增强的PatchMerging")
                else:
                    # 使用标准下采样层
                    merging_layer = PatchMerging(
                        input_resolution=resolution,
                        dim=dim,
                        norm_layer=norm_layer
                    )
                    swin_logger.info(f"层 {i_layer}: 使用标准PatchMerging")

                feature_bank_layers.append(merging_layer)
            else:
                feature_bank_layers.append(None)  # 最后一层没有下采样层

        return feature_bank_layers

    def _create_encoder_layers(self, patches_resolution, embed_dim, depths, num_heads,
                              window_size, dpr, norm_layer, drop_rate, attn_drop_rate,
                              qkv_bias, qk_scale, use_checkpoint):
        """创建编码器层"""
        layers = nn.ModuleList()
        count = 1

        for i_layer in range(self.num_layers):
            # 获取对应的下采样实例
            downsample_instance = self.feature_bank_layers[i_layer]

            # 创建BasicLayer，最后一层启用flag=True
            use_flag = (count == 4)
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                 patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample_instance,
                use_checkpoint=use_checkpoint,
                flag=use_flag
            )

            layers.append(layer)
            if not use_flag:
                count += 1

        return layers

    def _create_pbb_modules(self, embed_dim):
        """创建并行分支模块（ParallelBranchesBlock）"""
        pbb_list = nn.ModuleList()

        swin_logger.info("[Init] 创建并行分支模块 (ParallelBranchesBlock)")
        for i_layer in range(self.num_layers):
            if i_layer > 0:
                pbb = ParallelBranchesBlock(
                    in_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    out_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
                )
                pbb_list.append(pbb)

        return pbb_list

    def _create_decoder_layers(self, patches_resolution, embed_dim, depths, depths_decoder,
                              num_heads, window_size, dpr, norm_layer, drop_rate,
                              attn_drop_rate, qkv_bias, qk_scale, use_checkpoint):
        """创建解码器层"""
        layers_up = nn.ModuleList()
        concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            # 创建连接线性层
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            ) if i_layer > 0 else nn.Identity()

            # 创建上采样层
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                     patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer
                )
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                     patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint
                )

            layers_up.append(layer_up)
            concat_back_dim.append(concat_linear)

        return layers_up, concat_back_dim

    def _init_upsampling_modules(self, img_size, patch_size, embed_dim, norm_layer):
        """根据use_enhanced_up_x4初始化上采样模块"""
        if self.use_enhanced_up_x4:
            # 创建增强版上采样模块
            swin_logger.info("[Init] 创建增强版上采样模块")

            # 第一次x2上采样模块
            self.up_x2_first = nn.ModuleDict({
                'expand': nn.Linear(embed_dim, 4 * embed_dim, bias=False),
                'norm': norm_layer(4 * embed_dim),
                'refine': nn.Linear(embed_dim, embed_dim),
                'attention': nn.Linear(embed_dim, embed_dim),
                'gate': nn.Linear(embed_dim, 1)
            })

            # 第二次x2上采样模块
            self.up_x2_second = nn.ModuleDict({
                'expand': nn.Linear(embed_dim, 4 * embed_dim, bias=False),
                'norm': norm_layer(4 * embed_dim),
                'refine': nn.Linear(embed_dim, embed_dim),
                'attention': nn.Linear(embed_dim, embed_dim),
                'gate': nn.Linear(embed_dim, 1)
            })

            # 特征精炼模块
            self.feature_refine = nn.ModuleDict({
                'conv1': nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                'bn1': nn.BatchNorm2d(embed_dim),
                'conv2': nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                'bn2': nn.BatchNorm2d(embed_dim),
                'attention': ECAAttention(kernel_size=3)
            })

            # 特征融合参数
            self.alpha = nn.Parameter(torch.ones(1) * 0.8)
            self.beta = nn.Parameter(torch.ones(1) * 0.2)

            # 直接上采样卷积
            self.direct_up_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)

            # 明确将原始版的up设为None
            self.up = None
        else:
            # 创建原始版上采样模块
            swin_logger.info("[Init] 创建原始版上采样模块")

            # 定义原始版一次性上采样模块
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim_scale=4,
                dim=embed_dim
            )

            # 将增强版模块明确设为None
            self.up_x2_first = None
            self.up_x2_second = None
            self.feature_refine = None
            self.alpha = None
            self.beta = None
            self.direct_up_conv = None

    def _init_unetpp_modules(self, embed_dim):
        """初始化UNet++跳跃连接模块"""
        self.my_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        nb_filter = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]  # [96, 192, 384, 768]
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[1], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])

    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        """编码器和瓶颈层的前向传播"""
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            B, L, C = x.shape
            H, W = int(math.sqrt(L)), int(math.sqrt(L))
            x_downsample.append(x.view(B, H, W, C))
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x_downsample.append(x)

        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = self.norm(x)  # B L C

        # UNet++特征提取
        x0_1 = self.conv0_1(
            torch.cat([x_downsample[0].permute(0, 3, 1, 2), self.my_up(x_downsample[1].permute(0, 3, 1, 2))], 1))
        x1_1 = self.conv1_1(
            torch.cat([x_downsample[1].permute(0, 3, 1, 2), self.my_up(x_downsample[2].permute(0, 3, 1, 2))], 1))
        x0_2 = self.conv0_2(torch.cat([x_downsample[0].permute(0, 3, 1, 2), x0_1, self.my_up(x1_1)], 1))

        x2_1 = self.conv2_1(
            torch.cat([x_downsample[2].permute(0, 3, 1, 2), self.my_up(x_downsample[4].permute(0, 3, 1, 2))], 1))
        x1_2 = self.conv1_2(torch.cat([x_downsample[1].permute(0, 3, 1, 2), x1_1, self.my_up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x_downsample[0].permute(0, 3, 1, 2), x0_1, x0_2, self.my_up(x1_2)], 1))

        x_downsample_new = []
        x_downsample_new.append(torch.flatten(x0_3, start_dim=2, end_dim=-1).permute(0, 2, 1))
        x_downsample_new.append(torch.flatten(x1_2, start_dim=2, end_dim=-1).permute(0, 2, 1))
        x_downsample_new.append(torch.flatten(x2_1, start_dim=2, end_dim=-1).permute(0, 2, 1))

        return x, x_downsample_new

    def forward_up_features(self, x, x_downsample):
        """解码器和跳跃连接的前向传播"""
        self.decode_list = []

        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # 第一层直接应用上采样
                x = layer_up(x)
            else:
                # 获取对应的跳过连接特征
                x_downsamp = x_downsample[3 - inx]
                B, N, C = x_downsamp.shape
                H = int(N ** 0.5)
                W = H

                # 将x转换为BCHW格式以用于PBB
                x_input_for_pbb = x.view(B, H, W, C).permute(0, 3, 1, 2)

                # 条件化应用PBB（如果启用）
                if self.use_pbb and self.pbb_list is not None:
                    # swin_logger.info(f"[Runtime] 应用PBB于解码层 {inx}")
                    # 应用并行分支模块
                    x_after_pbb = self.pbb_list[inx - 1](x_input_for_pbb)
                    # 将处理后的结果转换回BNC格式
                    x = x_after_pbb.permute(0, 2, 3, 1).view(B, H*W, C)
                else:
                    # swin_logger.info(f"[Runtime] 解码层 {inx} 跳过PBB")
                    # 将x_input_for_pbb转回BNC格式
                    x = x_input_for_pbb.permute(0, 2, 3, 1).view(B, H*W, C)

                # 特征连接和变换
                x = torch.cat([x, x_downsamp], -1)
                x = self.concat_back_dim[inx](x)

                # 应用Swin块/上采样
                x = layer_up(x)

            self.decode_list.append(x)

        x = self.norm_up(x)  # B L C
        return x

    def up_x4(self, x, x_bak):
        """4倍上采样实现（根据use_enhanced_up_x4选择不同的实现）"""
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.use_enhanced_up_x4 and self.final_upsample == "expand_first":
            # === 增强版渐进式上采样 ===
            # swin_logger.info("[Runtime] 使用增强版上采样模块")

            # 初始化当前特征
            current_x = x

            # 第一次x2上采样
            x_first_expanded = self.up_x2_first['expand'](current_x)
            x_first_expanded = self.up_x2_first['norm'](x_first_expanded)
            B_up, L_up, C_expanded = x_first_expanded.shape
            H_new, W_new = H*2, W*2

            # 处理维度不匹配情况
            if C_expanded == 4*C:
                x_first_expanded = x_first_expanded.view(B, H, W, 4*C)
                x_first_expanded = rearrange(x_first_expanded, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C)
            else:
                swin_logger.warning(f"警告: up_x4第一次扩展中的维度不匹配. 预期 {4*C}, 实际获得 {C_expanded}")
                x_first_expanded = F.interpolate(current_x.view(B, H, W, C).permute(0, 3, 1, 2),
                                               scale_factor=2, mode='bilinear').permute(0, 2, 3, 1).view(B, H_new*W_new, C)

            # 特征精炼
            x_first_refined = x_first_expanded.view(B, H_new*W_new, C)
            x_first_attention = torch.sigmoid(self.up_x2_first['attention'](x_first_refined))
            x_first_refined = self.up_x2_first['refine'](x_first_refined)
            x_first_refined = x_first_refined * x_first_attention

            # 添加门控机制
            x_first_gate = torch.sigmoid(self.up_x2_first['gate'](x_first_refined))
            x_first_refined = x_first_refined * x_first_gate

            # 应用第一次特征精炼
            x_first = x_first_refined.view(B, H_new, W_new, C).permute(0, 3, 1, 2)
            x_first = self.feature_refine['conv1'](x_first)
            x_first = self.feature_refine['bn1'](x_first)
            x_first = F.leaky_relu(x_first, negative_slope=0.1)
            x_first = self.feature_refine['attention'](x_first)
            x_first = x_first.permute(0, 2, 3, 1).reshape(B, H_new*W_new, C)

            # 第二次x2上采样
            current_x = x_first
            x_second_expanded = self.up_x2_second['expand'](current_x)
            x_second_expanded = self.up_x2_second['norm'](x_second_expanded)
            B_up2, L_up2, C_expanded2 = x_second_expanded.shape
            H_final, W_final = H_new*2, W_new*2

            # 处理维度不匹配情况
            if C_expanded2 == 4*C:
                x_second_expanded = x_second_expanded.view(B, H_new, W_new, 4*C)
                x_second_expanded = rearrange(x_second_expanded, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C)
            else:
                swin_logger.warning(f"警告: up_x4第二次扩展中的维度不匹配. 预期 {4*C}, 实际获得 {C_expanded2}")
                x_second_expanded = F.interpolate(current_x.view(B, H_new, W_new, C).permute(0, 3, 1, 2),
                                                scale_factor=2, mode='bilinear').permute(0, 2, 3, 1).view(B, H_final*W_final, C)

            # 特征精炼
            x_second_refined = x_second_expanded.view(B, H_final*W_final, C)
            x_second_attention = torch.sigmoid(self.up_x2_second['attention'](x_second_refined))
            x_second_refined = self.up_x2_second['refine'](x_second_refined)
            x_second_refined = x_second_refined * x_second_attention

            # 添加门控机制
            x_second_gate = torch.sigmoid(self.up_x2_second['gate'](x_second_refined))
            x_second_refined = x_second_refined * x_second_gate

            # 应用第二次特征精炼
            x_second = x_second_refined.view(B, H_final, W_final, C).permute(0, 3, 1, 2)
            x_second = self.feature_refine['conv2'](x_second)
            x_second = self.feature_refine['bn2'](x_second)
            x_second = F.leaky_relu(x_second, negative_slope=0.1)
            x_second = self.feature_refine['attention'](x_second)

            # 使用progressive路径的结果
            x = x_second

        elif not self.use_enhanced_up_x4 and self.final_upsample == "expand_first":
            # === 原始版一次性上采样 ===
            # swin_logger.info("[Runtime] 使用原始版上采样模块")

            # 确保模块已正确初始化
            if self.up is None or self.output is None:
                raise RuntimeError("Original up_x4 modules not initialized properly when use_enhanced_up_x4 is False.")

            # 直接4倍上采样
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)

        else:
            # 处理配置错误或不支持的情况
            if self.final_upsample != "expand_first":
                raise ValueError(f"Unsupported final_upsample mode: {self.final_upsample}")

            # 默认使用原始逻辑
            swin_logger.info(f"警告: up_x4中出现意外状态. 默认使用原始逻辑.")
            if self.up is None or self.output is None:
                raise RuntimeError("Original up_x4 modules not initialized properly for fallback.")

            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)

        # 最终输出层（两种模式都需要）
        x = self.output(x)
        return x

    def enhance_features(self, x):
        """特征增强模块，添加空间和通道注意力"""
        # 空间特征增强
        enhanced = self.enhance_conv(x)
        enhanced = self.enhance_bn(enhanced)
        enhanced = F.leaky_relu(enhanced, negative_slope=0.1)

        # 添加注意力机制
        avg_pool = F.avg_pool2d(enhanced, kernel_size=7, stride=1, padding=3)
        max_pool = F.max_pool2d(enhanced, kernel_size=7, stride=1, padding=3)

        # 空间注意力
        spatial_attention = torch.sigmoid(avg_pool + max_pool)

        # 通道注意力
        channel_attention = torch.sigmoid(torch.mean(enhanced, dim=[2, 3], keepdim=True))

        # 应用注意力
        enhanced = enhanced * spatial_attention * channel_attention

        # 残差连接
        return x + enhanced * self.enhanced_feature_scale

    def forward(self, x):
        """模型完整前向传播"""
        x_bak = x
        # 编码器和瓶颈
        x, x_downsample = self.forward_features(x)
        # 解码器和跳过连接
        x = self.forward_up_features(x, x_downsample)
        # 最终上采样和输出
        x = self.up_x4(x, x_bak)
        return x

    def flops(self):
        """计算模型FLOPs（浮点运算次数）"""
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class FeatureBank(nn.Module):
    def __init__(self, feature_dim, num_clusters=None, queue_size=None, momentum=0.75, temperature=0.07, device=None):
        """
        特征银行模块，用于保存和更新聚类中心，以及在推理时用于特征融合
        优化版本：降低显存占用，加速计算

        Args:
            feature_dim (int): 特征向量的维度
            num_clusters (int, optional): 聚类中心的数量，如果为None则根据特征维度自适应设置
            queue_size (int, optional): 特征队列的大小，如果为None则根据特征维度自适应设置
            momentum (float): 聚类中心更新的动量因子
            temperature (float): 温度系数，用于调整相似度分布的平滑程度
            device (int, optional): GPU设备ID，如果为None则自动获取当前设备
        """
        super(FeatureBank, self).__init__()
        self.feature_dim = feature_dim

        # 设置GPU设备，支持动态设置
        if device is not None:
            # 使用传入的设备ID
            self.device = device
        elif torch.cuda.is_available():
            # 如果没有指定，则使用当前CUDA设备
            self.device = torch.cuda.current_device()
        else:
            # 默认为0
            self.device = 0

        # 优化1: 针对小目标的分割任务，使用更多聚类中心
        self.num_clusters = num_clusters or max(32, min(feature_dim // 8, 128))  # 增加聚类中心数量，提高小目标特征捕获能力

        # 优化2: 使用足够大的队列，确保特征多样性
        self.queue_size = queue_size or max(2000, min(feature_dim * 16, 6000))  # 扩大队列容量，存储更多样本特征

        self.momentum = momentum
        self.temperature = temperature

        # 修改: 统一使用float32存储以避免类型不匹配问题
        self.register_buffer("cluster_centers", torch.zeros(self.num_clusters, feature_dim, dtype=torch.float32))
        self.register_buffer("feature_queue", torch.zeros(self.queue_size, feature_dim, dtype=torch.float32))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("cluster_usage", torch.zeros(self.num_clusters, dtype=torch.float32))
        self.register_buffer("cluster_reliability", torch.zeros(self.num_clusters, dtype=torch.float32))

        # 优化4: 特征维度降维投影器 - 进一步降低维度以加速计算
        self.use_dim_reduction = True
        self.reduced_dim = min(256, feature_dim // 2)  # 保留更多特征维度
        if self.use_dim_reduction:
            self.dim_reducer = nn.Linear(feature_dim, self.reduced_dim, bias=False)
            # 使用正交初始化以保留距离关系
            nn.init.orthogonal_(self.dim_reducer.weight)
            self.register_buffer("reduced_centers", torch.zeros(self.num_clusters, self.reduced_dim, dtype=torch.float32))

        # 初始化标志
        self.initialized = False
        self.update_count = 0

        # 优化5: 添加计算缓存
        self.similarity_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        # 优化6: 批量更新控制 - 适应小目标
        self.update_interval = 20  # 降低更新间隔，更快更新特征
        self.min_updates_for_use = 2  # 保持启动条件低，更快开始工作

        # 区域划分和重要性阈值 - 小目标优化
        self.use_region_partition = True  # 使用区域划分优化
        self.region_size = 1  # 最小区域大小，更精细捕获小目标
        self.importance_threshold = 0.5  # 大幅降低重要性阈值，更容易发现小目标
        self.min_similarity_threshold = 0.4  # 降低相似度阈值，更容易匹配小目标特征

        # 聚类中心活跃度跟踪和自动清理
        self.activity_decay = 0.9  # 降低活跃度衰减因子，加快聚类中心更新
        self.inactivity_threshold = 5.0  # 降低不活跃阈值，更积极清理无用聚类中心
        self.register_buffer("cluster_activity", torch.zeros(self.num_clusters, dtype=torch.float32))

        # 强制使用float32，避免混合精度导致的问题
        self.use_mixed_precision = True

        # 新增优化: 聚类计算加速 - 启用FAISS
        self.use_faiss_clustering = True  # 使用FAISS库加速聚类
        self.max_cluster_compute_tokens = 4096  # 大幅提高处理token数量，获取更全面的聚类

        # 新增: FAISS-GPU支持
        self.use_faiss_gpu = True  # 优先使用GPU版本的FAISS
        self.gpu_resources = None  # 用于存储GPU资源句柄

        # 尝试初始化FAISS GPU资源 - 增强错误处理
        try:
            if self.use_faiss_gpu and torch.cuda.is_available():
                try:
                    # 先检查CUDA设备
                    device_count = torch.cuda.device_count()
                    if device_count == 0:
                        raise RuntimeError("没有可用的CUDA设备")

                    # 确保当前CUDA设备工作正常
                    device_id = torch.cuda.current_device()
                    device_name = torch.cuda.get_device_name(device_id)
                    swin_logger.info(f"当前使用CUDA设备 #{device_id}: {device_name}")

                    # 创建GPU资源，使用当前设备
                    self.gpu_resources = faiss.StandardGpuResources()

                    # 检查资源是否正确初始化
                    if self.gpu_resources is None:
                        raise RuntimeError("FAISS GPU资源创建失败")

                    swin_logger.info(f"成功初始化FAISS-GPU资源(设备 #{device_id})，将使用GPU加速聚类和检索")
                except Exception as gpu_err:
                    self.use_faiss_gpu = False
                    self.gpu_resources = None
                    swin_logger.warning(f"FAISS-GPU资源初始化失败: {gpu_err}，将使用CPU版本")
            else:
                self.use_faiss_gpu = False
                swin_logger.info("未检测到GPU或未启用FAISS-GPU，将使用CPU版本的FAISS")
        except ImportError:
            self.use_faiss_gpu = False
            swin_logger.warning("未安装FAISS库，将回退到PyTorch实现")
        except Exception as e:
            self.use_faiss_gpu = False
            swin_logger.warning(f"FAISS初始化过程中发生未知错误: {e}，将回退到PyTorch实现")

        # 新增优化: 分级特征选择 (只处理显著特征)
        self.use_feature_selection = True  # 使用特征选择
        self.feature_importance_threshold = 0.4  # 降低特征重要性阈值，保留更多小目标特征

        # 内存管理优化: 缓存大小限制和定期清理
        self.max_cache_size = 100  # 增加缓存条目数，提高命中率

        swin_logger.info(f"初始化高效版特征银行(小目标优化): 特征维度={feature_dim}, 聚类数={self.num_clusters}, 队列大小={self.queue_size}, 降维={self.reduced_dim}")

    def _reset_cache(self):
        """重置相似度缓存"""
        self.similarity_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    def _kmeans_update(self, features):
        """优化的K-means算法更新聚类中心，使用批处理和早停策略"""
        device = features.device

        # 检查缓存大小并在必要时清理
        if len(self.similarity_cache) > self.max_cache_size:
            self._reset_cache()

        # 统一使用float32计算
        features = features.float()

        # 新增优化: 更激进的特征选择 - 只使用重要的特征进行聚类更新
        if self.use_feature_selection:
            # 计算特征的L2范数作为重要性度量
            feature_norms = torch.norm(features, dim=1)
            # 归一化范数
            if feature_norms.max() > feature_norms.min():
                norm_max = feature_norms.max()
                norm_min = feature_norms.min()
                normalized_norms = (feature_norms - norm_min) / (norm_max - norm_min + 1e-6)
                # 选择重要性高于阈值的特征
                # 降低初始阈值以确保有足够的样本
                initial_threshold = max(0.3, self.feature_importance_threshold)
                important_indices = (normalized_norms > initial_threshold)

                # 如果样本太少，降低阈值直到满足最低要求
                min_required_samples = max(self.num_clusters * 4, 32)
                current_threshold = initial_threshold

                while important_indices.sum() < min_required_samples and current_threshold > 0.1:
                    current_threshold -= 0.05
                    important_indices = (normalized_norms > current_threshold)

                if important_indices.sum() >= max(8, self.num_clusters // 2):
                    features = features[important_indices]
                    swin_logger.info(f"特征选择：从{feature_norms.size(0)}个样本中选择了{important_indices.sum()}个重要样本进行聚类")

        # 优化: 限制用于聚类的特征数量
        if features.size(0) > self.max_cluster_compute_tokens:
            indices = torch.randperm(features.size(0), device=device)[:self.max_cluster_compute_tokens]
            features = features[indices]

        # 动态调整聚类中心数量，确保样本足够
        actual_num_clusters = self.num_clusters
        # FAISS建议样本数至少是聚类中心数的40倍
        if features.size(0) < self.num_clusters * 4:
            # 动态减少聚类中心数量，确保至少是样本数的1/4
            actual_num_clusters = max(4, features.size(0) // 4)
            swin_logger.info(f"样本不足，动态调整聚类中心数量: {self.num_clusters} -> {actual_num_clusters}")

        if not self.initialized:
            # 初始化聚类中心
            if features.size(0) >= actual_num_clusters:
                indices = torch.randperm(features.size(0), device=device)[:actual_num_clusters]
                # 初始化所有聚类中心
                self.cluster_centers.data.copy_(torch.zeros_like(self.cluster_centers))
                # 只设置活跃的聚类中心
                self.cluster_centers[:actual_num_clusters].data.copy_(features[indices].float())
            else:
                repeat_factor = math.ceil(actual_num_clusters / features.size(0))
                repeated_features = features.repeat(repeat_factor, 1)
                indices = torch.randperm(repeated_features.size(0), device=device)[:actual_num_clusters]
                # 初始化所有聚类中心
                self.cluster_centers.data.copy_(torch.zeros_like(self.cluster_centers))
                # 只设置活跃的聚类中心
                self.cluster_centers[:actual_num_clusters].data.copy_(repeated_features[indices].float())

            # 如果使用维度降维，初始化降维后的聚类中心
            if self.use_dim_reduction:
                with torch.no_grad():
                    self.reduced_centers.data.copy_(self.dim_reducer(self.cluster_centers.float()).float())

            # 初始化活跃度
            self.cluster_activity.data.fill_(0.1)  # 默认给低活跃度
            self.cluster_activity[:actual_num_clusters].fill_(1.0)  # 只有活跃的聚类中心给高活跃度

            self.initialized = True
            self._reset_cache()  # 重置缓存
            swin_logger.info(f"特征银行初始化完成: 特征维度={self.feature_dim}, 实际聚类数={actual_num_clusters}/{self.num_clusters}")
            return

        # 始终使用 PyTorch 实现进行聚类更新
        swin_logger.info("强制使用 PyTorch 实现进行聚类更新")

        # 确保数据在正确的设备上
        centers = self.cluster_centers.float().to(device)

        # 并行批处理以减少内存占用
        batch_size = 256
        num_batches = (features.size(0) + batch_size - 1) // batch_size

        # 用于收集每个聚类的样本总和和计数
        cluster_sums = torch.zeros_like(centers)
        cluster_counts = torch.zeros(self.num_clusters, device=device)

        # 更新活跃度 - 衰减所有聚类中心的活跃度
        self.cluster_activity.data.copy_((self.cluster_activity * self.activity_decay).float())

        # 并行处理批次
        import concurrent.futures
        from threading import Lock

        # 创建共享锁
        lock = Lock()

        def process_batch(start_idx, end_idx):
            batch_features = features[start_idx:end_idx]

            # 统一使用float32计算
            batch_features = batch_features.float()

            # 计算批次特征与聚类中心的距离
            # 使用矩阵乘法代替cdist计算余弦相似度，更快
            batch_features_norm = F.normalize(batch_features, dim=1, eps=1e-8)

            # 只考虑活跃的聚类中心
            active_centers = centers[:actual_num_clusters]
            centers_norm = F.normalize(active_centers, dim=1, eps=1e-8)

            # 计算余弦相似度
            similarity = torch.mm(batch_features_norm, centers_norm.t())
            distances = 1.0 - similarity

            # 分配到最近的聚类中心
            batch_assignments = torch.argmin(distances, dim=1)

            # 收集每个批次的结果 - 向量化实现
            local_cluster_sums = torch.zeros_like(centers)
            local_cluster_counts = torch.zeros(self.num_clusters, device=device)

            # 使用one-hot编码和矩阵乘法一次性计算所有聚类总和
            if actual_num_clusters > 0:
                # 创建one-hot编码
                one_hot = F.one_hot(batch_assignments, num_classes=actual_num_clusters).float()

                # 计算每个聚类的样本数
                cluster_sizes = one_hot.sum(dim=0)
                local_cluster_counts[:actual_num_clusters] = cluster_sizes

                # 使用矩阵乘法计算每个聚类的特征总和
                # 转置one-hot以便可以进行矩阵乘法: [num_clusters, batch_size] @ [batch_size, feature_dim]
                cluster_sums = torch.mm(one_hot.t(), batch_features)
                local_cluster_sums[:actual_num_clusters] = cluster_sums

            return local_cluster_sums, local_cluster_counts

        # 使用多线程并行处理
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, features.size(0))
                    futures.append(executor.submit(process_batch, start_idx, end_idx))

                # 收集结果
                for future in futures:
                    local_sums, local_counts = future.result()
                    with lock:
                        cluster_sums += local_sums
                        cluster_counts += local_counts
        except Exception as e:
            swin_logger.warning(f"并行处理失败: {e}，回退到串行处理")
            # 如果并行失败，回退到串行处理
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, features.size(0))
                local_sums, local_counts = process_batch(start_idx, end_idx)
                cluster_sums += local_sums
                cluster_counts += local_counts

        # 检测不活跃的聚类中心并重置
        inactive_mask = (self.cluster_activity < self.inactivity_threshold) & (cluster_counts == 0)
        num_inactive = inactive_mask.sum().item()

        if num_inactive > 0:
            # 找到最活跃的聚类中心
            _, most_active_indices = torch.topk(cluster_counts, k=min(num_inactive, cluster_counts.size(0)))

            # 对不活跃的聚类中心重新初始化 - 复制活跃中心并添加噪声
            for i, is_inactive in enumerate(inactive_mask):
                # 只处理前actual_num_clusters中的不活跃聚类中心
                if is_inactive and i < actual_num_clusters:
                    # 从活跃中心中随机选一个
                    if most_active_indices.size(0) > 0:
                        src_idx = most_active_indices[torch.randint(0, most_active_indices.size(0), (1,), device=device)]
                        # 复制并添加噪声
                        noise = torch.randn_like(centers[src_idx]) * 0.1
                        centers[i] = centers[src_idx] + noise
                        # 重置活跃度
                        self.cluster_activity[i] = 1.0

            swin_logger.info(f"重置了{num_inactive}个不活跃的聚类中心")

        # 更新聚类中心
        for c in range(self.num_clusters):
            # 只更新活跃的聚类中心
            if c < actual_num_clusters and cluster_counts[c] > 0:
                new_center = cluster_sums[c] / (cluster_counts[c] + 1e-8)
                old_center = centers[c]

                # 动量更新
                updated_center = self.momentum * old_center + (1 - self.momentum) * new_center
                centers[c] = updated_center

                # 更新可靠性指标
                reliability = min(1.0, cluster_counts[c] / 100.0)  # 归一化
                old_reliability = self.cluster_reliability[c].item()
                new_reliability = self.momentum * old_reliability + (1 - self.momentum) * reliability
                self.cluster_reliability[c] = torch.tensor([new_reliability], device=device).float()

                # 设置高活跃度
                self.cluster_activity[c] = 1.0
            elif c >= actual_num_clusters:
                # 非活跃聚类中心设置低活跃度
                self.cluster_activity[c] = 0.1

        # 更新存储的聚类中心
        self.cluster_centers.data.copy_(centers.float())

        # 如果使用维度降维，更新降维后的聚类中心
        if self.use_dim_reduction:
            with torch.no_grad():
                self.reduced_centers.data.copy_(self.dim_reducer(centers).float())

        # 重置缓存，因为聚类中心已更新
        self._reset_cache()
        swin_logger.info(f"PyTorch实现完成聚类更新: 实际使用聚类中心 {actual_num_clusters}/{self.num_clusters}")

    def update_queue(self, features):
        """优化的特征队列更新方法"""
        if features.size(1) != self.feature_dim:
            swin_logger.warning(f"特征维度不匹配: 输入={features.size(1)}, 预期={self.feature_dim}")
            return

        # 统一使用float32计算
        features = features.float()

        # 优化: 更激进地使用随机采样减少存储数量
        if features.size(0) > 16:
            # 随机采样16个样本减少计算量
            indices = torch.randperm(features.size(0), device=features.device)[:16]
            features = features[indices]

        batch_size = features.size(0)
        ptr = int(self.queue_ptr)
        device = features.device

        # 确保队列在正确的设备上
        if self.feature_queue.device != device:
            self.feature_queue = self.feature_queue.to(device)
            self.queue_ptr = self.queue_ptr.to(device)

        try:
            # 使用float32计算
            features_float = features.float()

            # 使用CUDA异步操作更新队列
            with torch.cuda.stream(torch.cuda.Stream()) if torch.cuda.is_available() else nullcontext():
                # 将新特征添加到队列
                if ptr + batch_size <= self.queue_size:
                    self.feature_queue[ptr:ptr + batch_size].copy_(features_float, non_blocking=True)
                else:
                    remain = self.queue_size - ptr
                    self.feature_queue[ptr:].copy_(features_float[:remain], non_blocking=True)
                    if batch_size - remain > 0:
                        self.feature_queue[:batch_size - remain].copy_(features_float[remain:], non_blocking=True)
        except Exception as e:
            swin_logger.warning(f"特征银行队列更新失败: {e}")

        # 更新指针
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

        # 大幅降低聚类更新频率以减少计算
        if not self.initialized or (self.update_count % self.update_interval == 0):
            try:
                # 使用已填充的部分队列，而不是全部队列
                filled_size = min(self.update_count * batch_size, self.queue_size)
                active_queue = self.feature_queue[:filled_size]

                # 确保有足够的样本进行聚类
                if active_queue.size(0) >= max(self.num_clusters // 2, 32):
                    # 在单独的CUDA流中进行聚类计算
                    if torch.cuda.is_available():
                        stream = torch.cuda.Stream()
                        with torch.cuda.stream(stream):
                            self._kmeans_update(active_queue)
                        # 不等待聚类完成，异步进行
                    else:
                        self._kmeans_update(active_queue)
            except Exception as e:
                swin_logger.warning(f"聚类更新失败: {str(e)}")

        self.update_count += 1

    def get_topk_clusters(self, features, k=5):
        """优化的TopK聚类中心检索方法"""
        device = features.device
        batch_size = features.size(0)

        # 如果特征银行未初始化或更新次数不足，返回虚拟结果
        if not self.initialized or self.update_count < self.min_updates_for_use:
            dummy_idx = torch.zeros(batch_size, min(k, self.num_clusters), dtype=torch.long, device=device)
            dummy_sim = torch.zeros(batch_size, min(k, self.num_clusters), device=device)
            return dummy_idx, dummy_sim

        # 统一使用float32计算
        features_compute = features.float()

        # 稀疏计算 - 只处理特征norm最大的前N%特征
        if self.use_feature_selection and features.size(0) > 10:
            # 计算特征重要性
            feature_norms = torch.norm(features_compute, dim=1)

            # 只处理前30%最重要的特征
            k_important = max(3, int(features.size(0) * 0.3))
            _, important_indices = torch.topk(feature_norms, k_important)
            important_features = features_compute[important_indices]

            # 为非重要特征创建虚拟结果
            dummy_indices = torch.zeros(batch_size - k_important, min(k, self.num_clusters),
                                        dtype=torch.long, device=device)
            dummy_sim = torch.zeros(batch_size - k_important, min(k, self.num_clusters), device=device)

            # 仅对重要特征计算最近的聚类中心
            topk_idx_important, topk_sim_important = self._compute_topk(important_features, k)

            # 合并重要和非重要特征的结果
            all_indices = torch.zeros(batch_size, min(k, self.num_clusters), dtype=torch.long, device=device)
            all_sim = torch.zeros(batch_size, min(k, self.num_clusters), device=device)

            # 放回原来的位置
            all_indices[important_indices] = topk_idx_important
            all_sim[important_indices] = topk_sim_important

            # 填充非重要特征位置
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            mask[important_indices] = False
            non_important_indices = torch.nonzero(mask).squeeze(1)

            if non_important_indices.numel() > 0:
                all_indices[non_important_indices] = dummy_indices
                all_sim[non_important_indices] = dummy_sim

            return all_indices, all_sim

        # 检查缓存
        batch_hash = hash(features_compute.sum().item())
        if batch_hash in self.similarity_cache:
            self.cache_hit_count += 1
            return self.similarity_cache[batch_hash]

        self.cache_miss_count += 1

        # 对所有特征计算最近的聚类中心
        topk_idx, topk_sim = self._compute_topk(features_compute, k)

        # 缓存结果
        if len(self.similarity_cache) < self.max_cache_size:
            self.similarity_cache[batch_hash] = (topk_idx, topk_sim)

        return topk_idx, topk_sim

    def _compute_topk(self, features, k=5):
        """计算特征与聚类中心的TopK相似度，优化版本"""
        device = features.device
        batch_size = features.size(0)

        # 确保所有数据在同一设备上
        if self.cluster_centers.device != device:
            self.cluster_centers = self.cluster_centers.to(device)
            if self.use_dim_reduction:
                self.reduced_centers = self.reduced_centers.to(device)
            self.cluster_activity = self.cluster_activity.to(device)

        # 统一使用float32计算
        features = features.float()

        # 只考虑活跃的聚类中心 (活跃度 >= 0.5)
        active_mask = self.cluster_activity >= 0.5
        active_indices = torch.nonzero(active_mask).squeeze(1)
        num_active = active_indices.size(0)

        if num_active == 0:
            # 如果没有活跃的聚类中心，使用前4
            num_active = min(4, self.num_clusters)
            active_indices = torch.arange(num_active, device=device)

        # 调整k值，确保不超过活跃聚类中心数量
        k = min(k, num_active)

        # 尝试使用更快的FAISS库计算最近邻
        if self.use_faiss_clustering and features.size(0) > 32:
            try:
                import faiss
                import numpy as np

                # 优化: 对于非常大的批次，使用随机采样进行加速
                if batch_size > 512:
                    sample_size = min(batch_size, 256)  # 最多采样256个样本
                    sample_indices = torch.randperm(batch_size, device=device)[:sample_size]
                    features_sampled = features[sample_indices]
                else:
                    features_sampled = features
                    sample_indices = None

                # 后续FAISS计算使用采样后的特征
                features_np = features_sampled.float().detach().cpu().numpy()

                # 只使用活跃的聚类中心
                centers = self.cluster_centers[active_indices]
                centers_np = centers.float().detach().cpu().numpy()

                # 确保数据类型是float32
                features_np = features_np.astype(np.float32)
                centers_np = centers_np.astype(np.float32)

                # 创建FAISS索引 - 使用更快的索引类型
                if centers_np.shape[0] > 1000:
                    # 对大型聚类中心集合使用IVF索引
                    nlist = min(int(np.sqrt(centers_np.shape[0])), 30)
                    quantizer = faiss.IndexFlatIP(self.feature_dim)
                    index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist)
                    index.train(centers_np)
                else:
                    # 对小型集合使用简单的Flat索引
                    index = faiss.IndexFlatIP(self.feature_dim)

                # 如果GPU资源可用，使用GPU加速
                if self.use_faiss_gpu and self.gpu_resources is not None:
                    index = faiss.index_cpu_to_gpu(self.gpu_resources, self.device, index)
                    # swin_logger.info("[特征匹配] 使用GPU加速的FAISS进行相似度检索")

                # 归一化特征和中心以计算余弦相似度
                faiss.normalize_L2(features_np)
                faiss.normalize_L2(centers_np)

                # 添加中心到索引
                index.add(centers_np)

                # 搜索最相似的k个中心
                sim, idx = index.search(features_np, k)

                # 如果使用了采样，将结果扩展回原始大小
                if sample_indices is not None:
                    # 创建完整大小的结果张量
                    full_sim = torch.zeros(batch_size, k, device=device)
                    full_idx = torch.zeros(batch_size, k, dtype=torch.long, device=device)

                    # 将采样结果复制到对应位置
                    topk_sim_sampled = torch.from_numpy(sim).to(device)
                    topk_idx_sampled = torch.from_numpy(idx).to(device)

                    # 填充完整结果
                    full_sim[sample_indices] = topk_sim_sampled
                    full_idx[sample_indices] = topk_idx_sampled

                    # 填充未采样位置 - 使用最近的已采样结果
                    if batch_size > sample_size:
                        # 计算每个未采样点到采样点的距离
                        unsampled_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                        unsampled_mask[sample_indices] = False
                        unsampled_indices = torch.nonzero(unsampled_mask).squeeze(1)

                        # 使用最相似的采样点结果
                        for i in range(0, len(unsampled_indices), 64):  # 分批处理
                            end_i = min(i + 64, len(unsampled_indices))
                            current_unsampled = unsampled_indices[i:end_i]
                            current_features = features[current_unsampled]

                            # 找到最相似的采样点
                            current_features_norm = F.normalize(current_features, dim=1, eps=1e-8)
                            sampled_features_norm = F.normalize(features_sampled, dim=1, eps=1e-8)

                            # 计算余弦相似度
                            similarity = torch.mm(current_features_norm, sampled_features_norm.t())
                            _, most_similar = torch.max(similarity, dim=1)

                            # 复制最相似采样点的结果
                            full_sim[current_unsampled] = topk_sim_sampled[most_similar]
                            full_idx[current_unsampled] = topk_idx_sampled[most_similar]

                    topk_sim = full_sim
                    topk_idx_local = full_idx
                else:
                    # 直接使用完整结果
                    topk_sim = torch.from_numpy(sim).to(device)
                    topk_idx_local = torch.from_numpy(idx).to(device)

                # 将局部索引映射回全局索引
                # 创建有效范围掩码
                valid_mask = topk_idx_local < len(active_indices)

                # 创建映射张量
                topk_idx = torch.zeros_like(topk_idx_local)

                # 使用掩码索引，一次性处理有效位置
                if torch.any(valid_mask):
                    # 将active_indices转换为张量
                    active_indices_tensor = active_indices
                    # 创建映射
                    topk_idx[valid_mask] = active_indices_tensor[topk_idx_local[valid_mask]]

                return topk_idx, topk_sim

            except Exception as e:
                # 增强错误报告，确保即使异常消息为空也能捕获有用信息
                error_type = type(e).__name__
                error_msg = str(e) if str(e) else "空错误消息，可能是FAISS内部错误"
                error_details = f"类型:{error_type}, 消息:{error_msg}"

                # 添加特征大小信息和批次信息
                feature_info = f"特征数量:{features.size(0)}, 维度:{features.size(1)}"

                # 添加FAISS版本信息
                faiss_version = getattr(faiss, '__version__', "未知")

                swin_logger.warning(f"FAISS相似度计算失败 - {error_details} ({feature_info}, FAISS:{faiss_version})，回退到PyTorch实现")
                # 如果FAISS失败，回退到PyTorch实现
                pass

        # PyTorch实现 - 优化版本 (当上述方法失败或未启用时使用)
        # 优化: 分批处理以减少内存需求
        max_batch_size = 128  # 最大批次大小

        if batch_size > max_batch_size:
            # 分批次处理
            topk_sim_list = []
            topk_idx_list = []

            for i in range(0, batch_size, max_batch_size):
                end_i = min(i + max_batch_size, batch_size)
                batch_features = features[i:end_i]

                # 归一化特征
                batch_features_norm = F.normalize(batch_features, dim=1, eps=1e-8)

                # 获取活跃的聚类中心
                centers = self.cluster_centers[active_indices]
                centers_norm = F.normalize(centers, dim=1, eps=1e-8)

                # 计算相似度矩阵 - 余弦相似度
                batch_similarity = torch.mm(batch_features_norm, centers_norm.t())

                # 获取TopK
                batch_topk_sim, batch_topk_idx_local = torch.topk(batch_similarity, k, dim=1)

                # 将局部索引映射回全局索引
                batch_topk_idx = torch.zeros_like(batch_topk_idx_local)
                valid_mask = batch_topk_idx_local < len(active_indices)
                if torch.any(valid_mask):
                    batch_topk_idx[valid_mask] = active_indices[batch_topk_idx_local[valid_mask]]

                topk_sim_list.append(batch_topk_sim)
                topk_idx_list.append(batch_topk_idx)

            # 合并结果
            topk_sim = torch.cat(topk_sim_list, dim=0)
            topk_idx = torch.cat(topk_idx_list, dim=0)
        else:
            # 单次处理小批量
            # 归一化特征和中心
            features_norm = F.normalize(features, dim=1, eps=1e-8)
            centers = self.cluster_centers[active_indices]
            centers_norm = F.normalize(centers, dim=1, eps=1e-8)

            # 计算余弦相似度
            similarity = torch.mm(features_norm, centers_norm.t())

            # 获取TopK结果
            topk_sim, topk_idx_local = torch.topk(similarity, k, dim=1)

            # 将局部索引映射回全局索引
            topk_idx = torch.zeros_like(topk_idx_local)
            valid_mask = topk_idx_local < len(active_indices)
            if torch.any(valid_mask):
                topk_idx[valid_mask] = active_indices[topk_idx_local[valid_mask]]

        # 滤除相似度过低的结果
        min_sim_mask = topk_sim < self.min_similarity_threshold
        topk_sim[min_sim_mask] = 0.0

        return topk_idx, topk_sim

    def fuse_features(self, features, topk_idx, topk_sim):
        """优化的特征融合方法"""
        device = features.device

        # 如果特征银行未初始化或更新次数不足，直接返回原始特征
        if not self.initialized or self.update_count < self.min_updates_for_use:
            return features

        batch_size = features.size(0)
        k = topk_idx.size(1)

        # 确保所有数据在同一设备上
        if self.cluster_centers.device != device:
            self.cluster_centers = self.cluster_centers.to(device)
            self.cluster_reliability = self.cluster_reliability.to(device)
            self.cluster_activity = self.cluster_activity.to(device)

        # 统一使用float32计算
        features_compute = features.float()

        # 优化12: 批处理获取聚类中心以减少内存
        batch_centers = []
        batch_reliability = []
        batch_activity = []

        try:
            # 优化：使用向量化操作替代双循环
            batch_centers = []
            batch_reliability = []
            batch_activity = []

            # 每次处理一个小批次，减少内存峰值
            mini_batch = 16  # 增大mini_batch以减少迭代次数
            for i in range(0, batch_size, mini_batch):
                end_i = min(i + mini_batch, batch_size)
                current_batch_size = end_i - i

                # 获取当前小批次的索引
                current_indices = topk_idx[i:end_i]  # [current_batch, top_k]

                # 批量获取中心、可靠性和活跃度
                # 重塑索引以便批量索引
                flat_indices = current_indices.reshape(-1)  # [current_batch*top_k]

                # 批量获取对应的数据
                flat_centers = self.cluster_centers[flat_indices].float()  # [current_batch*top_k, feature_dim]
                flat_reliability = self.cluster_reliability[flat_indices].float()  # [current_batch*top_k]
                flat_activity = self.cluster_activity[flat_indices].float()  # [current_batch*top_k]

                # 重塑回原始形状
                centers = flat_centers.reshape(current_batch_size, -1, flat_centers.size(-1))  # [current_batch, top_k, feature_dim]
                reliability = flat_reliability.reshape(current_batch_size, -1)  # [current_batch, top_k]
                activity = flat_activity.reshape(current_batch_size, -1)  # [current_batch, top_k]

                batch_centers.append(centers)
                batch_reliability.append(reliability)
                batch_activity.append(activity)

            # 合并小批次结果
            batch_centers = torch.cat(batch_centers, dim=0)
            batch_reliability = torch.cat(batch_reliability, dim=0)
            batch_activity = torch.cat(batch_activity, dim=0)
        except Exception as e:
            swin_logger.warning(f"特征银行操作失败: {str(e)}")
            # 操作失败时直接返回原始特征
            return features

        # 标准化可靠性和活跃度
        scaled_reliability = batch_reliability / (self.update_count + 1)

        # 归一化活跃度到[0.5, 1.0]范围
        activity_factor = 0.5 + 0.5 * (batch_activity / (batch_activity.max() + 1e-6))

        # 结合相似度、可靠性和活跃度计算权重
        combined_weights = topk_sim * torch.clamp(scaled_reliability, min=0.1) * activity_factor

        # 新增优化: 使用相似度阈值过滤低相似度的聚类中心
        if self.min_similarity_threshold > 0:
            # 对低于阈值的相似度显著降低权重
            sim_mask = (topk_sim < self.min_similarity_threshold)
            combined_weights[sim_mask] *= 0.1

        # 使用softmax将权重归一化
        weights = F.softmax(combined_weights / (self.temperature + 1e-8), dim=1).unsqueeze(2)

        # 优化13: 使用批量矩阵乘法替代循环，同时通过分批减少内存占用
        mini_batch = 32  # 增大mini_batch大小以提高效率
        weighted_centers = torch.zeros_like(features.float())

        # 使用更高效的torch.baddbmm操作计算加权和
        for i in range(0, batch_size, mini_batch):
            end_i = min(i + mini_batch, batch_size)
            current_batch_size = end_i - i

            # 使用批量矩阵乘法一次性计算当前批次的加权和
            weighted_sum = torch.bmm(
                weights[i:end_i].transpose(1, 2),  # [batch, 1, k]
                batch_centers[i:end_i]             # [batch, k, feat_dim]
            )

            # 移除多余的维度并存储结果
            weighted_centers[i:end_i] = weighted_sum.squeeze(1)

        # 计算自适应融合系数，动态调整权重
        max_weight = min(0.3, 0.3 * (self.update_count / (self.min_updates_for_use + 1000)))

        # 使用均匀融合或基于特征重要性的自适应融合
        if self.use_feature_selection:
            # 计算特征重要性 - 使用L2范数作为指标
            feature_norms = torch.norm(features.float(), dim=1, keepdim=True)
            # 归一化到[0.2, 1.0]范围，重要性越高融合权重越大
            if feature_norms.max() > feature_norms.min():
                importance_factor = 0.2 + 0.8 * (feature_norms - feature_norms.min()) / (feature_norms.max() - feature_norms.min() + 1e-8) # 添加eps
                fusion_weight = max_weight * importance_factor
            else:
                fusion_weight = torch.ones_like(feature_norms) * max_weight
        else:
            # 使用统一的融合权重
            fusion_weight = torch.ones_like(features[:,:1]) * max_weight

        # 最终特征 = 原始特征 * (1-权重) + 增强特征 * 权重
        enhanced_features = features * (1 - fusion_weight) + weighted_centers * fusion_weight

        return enhanced_features


class PatchMergingWithFeatureBank(nn.Module):
    r""" 带有特征银行的Patch合并层，优化版本

    当use_feature_bank=True时使用此类作为下采样层。

    与标准PatchMerging相比:
    - 更强大: 具有特征记忆和融合能力，可利用历史特征信息
    - 更适合处理复杂场景: 特别是对小目标、罕见特征和边界区域
    - 计算复杂度更高: 需要维护特征队列和聚类中心
    - 自适应性更强: 可以根据特征重要性动态调整处理流程

    Args:
        input_resolution (tuple[int]): 输入特征图的分辨率
        dim (int): 输入通道数
        norm_layer (nn.Module): 规范化层. 默认: nn.LayerNorm
        grid_sample_size (tuple[int]): 空间网格采样时每个维度的单元数 (例如 (4, 4)). 如果为 None, 则不使用网格采样.
        sample_per_grid (int): 每个网格单元采样多少个点.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm,
                 grid_sample_size: Optional[Tuple[int, int]] = (4, 4), # 改为初始化参数
                 sample_per_grid: int = 1): # 改为初始化参数
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        # 设置特征银行维度
        feature_dim = 2 * dim
        # 获取当前设备ID并传递给FeatureBank
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

        # 优化小目标分割的特征银行参数
        self.feature_bank = FeatureBank(
            feature_dim=feature_dim,       # 降低温度参数，增强特征区分度
            device=device_id
        )

        # 轻量级融合后规范化
        self.norm_fused = norm_layer(feature_dim)

        # 可学习融合权重，但初始值设为较小值优先使用原始特征
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.2)

        # 使用特征银行的控制标志
        self.use_feature_bank = True

        # 初始阶段不使用特征银行的训练步数（减少预热时间以更快开始使用特征银行）
        self.warmup_steps = 50
        self.step_count = 0

        # 区域处理优化参数
        self.use_region_based = True   # 启用区域处理
        self.importance_threshold = 0.35  # 降低区域重要性阈值，处理更多潜在小目标区域

        # 网格采样参数
        self.grid_sample_size = grid_sample_size
        self.sample_per_grid = sample_per_grid
        if self.grid_sample_size is not None:
             swin_logger.info(f"PatchMerging initializing with Spatial Grid Sampling: Grid Size={self.grid_sample_size}, Samples/Grid={self.sample_per_grid}")

    def _spatial_grid_sampling(self, features: torch.Tensor,
                                b_indices: torch.Tensor,
                                pos_indices: torch.Tensor,
                                H_down: int, W_down: int,
                                grid_size_h: int, grid_size_w: int,
                                sample_per_cell: int) -> torch.Tensor:
        """
        从重要特征中执行空间网格采样。

        Args:
            features: 重要区域的特征张量 [N_important, D].
            b_indices: 每个重要特征对应的批次索引 [N_important].
            pos_indices: 每个重要特征对应的扁平化位置索引 [N_important].
            H_down, W_down: 下采样后的特征图高度和宽度.
            grid_size_h, grid_size_w: 网格的高度和宽度单元数.
            sample_per_cell: 每个网格单元采样多少个点.

        Returns:
            采样后的特征张量 [N_sampled, D].
        """
        device = features.device
        sampled_features_list = []

        # 计算网格单元的实际像素尺寸
        # 处理无法整除的情况，确保所有像素都被覆盖
        cell_h = (H_down + grid_size_h - 1) // grid_size_h
        cell_w = (W_down + grid_size_w - 1) // grid_size_w

        # 遍历每个批次元素
        unique_batches = torch.unique(b_indices)
        for b in unique_batches:
            batch_mask = (b_indices == b)
            batch_features = features[batch_mask]
            batch_pos_indices = pos_indices[batch_mask]

            if batch_features.numel() == 0:
                continue

            # 将扁平化索引映射回 H, W 坐标
            h_coords = batch_pos_indices // W_down
            w_coords = batch_pos_indices % W_down

            # 计算每个特征所属的网格单元索引
            grid_h_indices = h_coords // cell_h
            grid_w_indices = w_coords // cell_w

            # 组合批次内网格索引，方便分组
            # 使用一个较大的基数避免冲突
            combined_grid_indices = grid_h_indices * grid_size_w + grid_w_indices

            # 遍历每个唯一的网格单元
            unique_cells = torch.unique(combined_grid_indices)
            for cell_idx in unique_cells:
                cell_mask = (combined_grid_indices == cell_idx)
                features_in_cell = batch_features[cell_mask]

                if features_in_cell.numel() == 0:
                    continue

                num_features_in_cell = features_in_cell.size(0)

                # 从单元格中采样
                if num_features_in_cell <= sample_per_cell:
                    # 如果单元格内特征数不足，全部选中
                    sampled_features_list.append(features_in_cell)
                else:
                    # 如果特征数充足，计算每个特征向量的 L2 范数（幅度）
                    try:
                        # 计算 L2 范数
                        feature_magnitudes = torch.norm(features_in_cell.float(), p=2, dim=1)
                        # 选择幅度最高的 top-k 个特征的索引
                        _, top_indices = torch.topk(feature_magnitudes, k=sample_per_cell)
                        # 根据索引选择特征
                        sampled_features_list.append(features_in_cell[top_indices])
                    except Exception as e:
                         # 如果计算范数或topk出错，回退到随机采样
                         swin_logger.warning(f"网格采样单元中的幅值计算/topk失败 (大小 {num_features_in_cell}): {e}. 回退到随机采样.")
                         perm = torch.randperm(num_features_in_cell, device=device)[:sample_per_cell]
                         sampled_features_list.append(features_in_cell[perm])

        if not sampled_features_list:
            return torch.empty(0, features.size(1), device=device, dtype=features.dtype)

        # 合并所有采样到的特征
        sampled_features = torch.cat(sampled_features_list, dim=0)
        return sampled_features

    def forward(self, x):
        """
        x: B, H*W, C
        优化的前向传播，降低内存占用，提高计算效率
        """

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征尺寸错误"
        assert H % 2 == 0 and W % 2 == 0, f"x尺寸({H}*{W})不是偶数。"

        device = x.device

        # 执行标准下采样操作
        x = x.view(B, H, W, C)

        # 减少内存重分配，直接使用视图操作
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        # 拼接操作
        x_merged = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x_reshaped = x_merged.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # 规范化和降维
        x_norm = self.norm(x_reshaped)
        x_reduced = self.reduction(x_norm)  # B H/2*W/2 2*C

        # 更新步数计数
        self.step_count += 1

        # 热身期间或评估模式时直接返回原始结果
        if not self.use_feature_bank or not self.training or self.step_count <= self.warmup_steps:
            return x_reduced

        # --- 特征银行逻辑开始 ---
        # === 特征银行模块 ===
        # 作用: 存储和利用历史特征信息，增强模型对稀疏或难以识别的特征的表达能力
        # 处理流程: 1. 提取特征向量 2. 区域重要性分析 3. 特征聚类更新 4. 特征融合
        # 该模块通过记忆和融合历史特征，显著提升对小目标和边界区域的分割性能
        # swin_logger.info(f"[调试] 在PatchMerging中执行特征银行逻辑 (步骤 {self.step_count}).") # <-- 添加打印

        try:

            # 提取特征向量进行更新，解决维度不匹配问题
            feature_vectors = x_reduced.reshape(-1, x_reduced.size(-1))  # [B*H*W/4, 2*C]

            # 统一使用float32进行计算
            feature_vectors = feature_vectors.float()

            # 新增优化: 如果启用区域处理，进行区域划分
            if self.use_region_based and H//2 >= 4 and W//2 >= 4:
                # 计算下采样后的H、W
                H_down, W_down = H//2, W//2

                # 计算特征幅值作为重要性指标
                feature_magnitude = torch.norm(x_reduced, dim=2)  # [B, H_down*W_down]
                feature_magnitude = feature_magnitude.view(B, H_down, W_down)

                # 计算每个区域的平均幅值
                region_size = min(4, min(H_down//2, W_down//2))  # 确保区域大小合理
                if region_size <= 0: # 避免 region_size 为 0 或负数
                     swin_logger.warning(f"计算的区域大小为 {region_size}. 跳过基于区域的处理.")
                     pass # 让代码继续执行到全局融合部分
                else:
                    num_regions_h = H_down // region_size
                    num_regions_w = W_down // region_size

                    if num_regions_h > 0 and num_regions_w > 0:
                        usable_h = num_regions_h * region_size
                        usable_w = num_regions_w * region_size

                        # 裁剪到可用大小
                        feature_magnitude_usable = feature_magnitude[:, :usable_h, :usable_w]

                        # 重塑以计算区域均值 [B, num_regions_h, region_size, num_regions_w, region_size]
                        regions = feature_magnitude_usable.reshape(
                            B, num_regions_h, region_size, num_regions_w, region_size
                        )

                        # 计算每个区域的平均幅值 [B, num_regions_h, num_regions_w]
                        region_importance = regions.mean(dim=[2, 4])

                        # 新的归一化逻辑，按批次处理
                        region_max_per_batch = region_importance.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1) # Shape [B, 1, 1]
                        region_min_per_batch = region_importance.view(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1) # Shape [B, 1, 1]
                        denominator = region_max_per_batch - region_min_per_batch + 1e-6
                        # 创建掩码，标记可以安全进行除法的地方 (max > min)
                        # 使用 .detach() 来创建掩码，避免影响梯度
                        safe_division_mask = (region_max_per_batch > region_min_per_batch).float() # Shape [B, 1, 1]
                        # 安全地计算归一化值，对于 max == min 的情况，结果会是 0 / eps
                        normalized_importance = (region_importance - region_min_per_batch) / denominator
                        # 应用掩码，使得 max == min 的批次元素结果为 0
                        region_importance = normalized_importance * safe_division_mask

                        # 创建区域掩码，只处理重要性高于阈值的区域
                        region_mask = (region_importance > self.importance_threshold)

                        # 扩展掩码到像素级别 - 向量化实现
                        pixel_mask = torch.zeros(B, H_down, W_down, device=device)

                        # 获取所有激活区域的索引
                        active_regions = torch.nonzero(region_mask, as_tuple=True)
                        b_idx_region, i_idx, j_idx = active_regions

                        # 计算每个区域对应的像素范围
                        h_start = i_idx * region_size
                        h_end = (i_idx + 1) * region_size
                        w_start = j_idx * region_size
                        w_end = (j_idx + 1) * region_size

                        # 使用索引操作设置激活区域的像素值
                        for idx in range(len(b_idx_region)):
                            pixel_mask[b_idx_region[idx],
                                      h_start[idx]:h_end[idx],
                                      w_start[idx]:w_end[idx]] = 1.0

                        # 将掩码调整为与 feature_vectors 兼容的形状
                        pixel_mask = pixel_mask.view(B, -1) # [B, H_down*W_down]

                        # 只对重要区域应用特征银行处理
                        important_indices = torch.nonzero(pixel_mask) # [N_important, 2] (col 0: batch_idx, col 1: pos_idx)

                        if important_indices.size(0) > 0:
                            # 提取重要区域的特征向量
                            b_indices = important_indices[:, 0]
                            pos_indices = important_indices[:, 1]

                            important_features = x_reduced[b_indices, pos_indices] # [N_important, D]
                            important_features = important_features.float()

                            # --- 使用空间网格采样替代 K-Means ---
                            update_features = torch.empty(0, important_features.size(1), device=device, dtype=important_features.dtype)
                            if self.grid_sample_size is not None and self.step_count % 4 == 0 and important_features.size(0) > 1:
                                grid_h, grid_w = self.grid_sample_size
                                if H_down >= grid_h and W_down >= grid_w: # 确保特征图大小至少等于网格大小
                                    try:
                                        update_features = self._spatial_grid_sampling(
                                            important_features.detach(), # 使用 detach 避免梯度计算
                                            b_indices,
                                            pos_indices,
                                            H_down, W_down,
                                            grid_h, grid_w,
                                            self.sample_per_grid
                                        )
                                        swin_logger.info(f"空间网格采样生成了 {update_features.shape[0]} 个特征用于队列更新.")
                                    except Exception as e:
                                        swin_logger.warning(f"空间网格采样失败: {e}")
                                else:
                                     swin_logger.warning(f"特征图大小 ({H_down}x{W_down}) 对于网格大小 {self.grid_sample_size} 过小. 跳过网格采样.")

                            # 更新特征队列 (如果采样到了特征)
                            if update_features.numel() > 0:
                                try:
                                    self.feature_bank.update_queue(update_features)
                                except Exception as e:
                                     swin_logger.warning(f"网格采样后特征库队列更新失败: {e}")

                            # --- 特征融合部分 (保持不变) ---
                            x_reduced_clone = x_reduced.clone()
                            try:
                                topk_idx, topk_sim = self.feature_bank.get_topk_clusters(important_features, k=3)
                                fused_important_features = self.feature_bank.fuse_features(important_features, topk_idx, topk_sim)
                                fused_important_features = fused_important_features.to(x_reduced.dtype)

                                # 将融合后的特征放回原位
                                x_reduced_clone[b_indices, pos_indices] = fused_important_features

                                # 应用可学习的融合权重
                                alpha = torch.clamp(self.fusion_weight, 0.0, 0.5)
                                x_reduced = (1 - alpha) * x_reduced + alpha * x_reduced_clone

                                return x_reduced # ****** 区域处理成功，直接返回 ******

                            except Exception as e:
                                swin_logger.warning(f"基于区域的特征融合失败: {e}. 回退到全局融合.")
                                pass # 继续执行下面的全局融合逻辑
                        else:
                             pass # 没有重要区域，继续执行全局融合
                    else:
                         pass # 区域划分无法进行，继续全局融合

            try:
                topk_idx, topk_sim = self.feature_bank.get_topk_clusters(feature_vectors, k=3)
            except Exception as e:
                swin_logger.warning(f"全局获取聚类中心失败: {e}")
                # 如果获取聚类中心失败，直接返回原始特征
                return x_reduced

            # 融合特征
            try:
                fused_vectors = self.feature_bank.fuse_features(feature_vectors, topk_idx, topk_sim)
            except Exception as e:
                swin_logger.warning(f"特征融合失败: {e}")

                return x_reduced

            # 将融合后的特征向量重新调整为原始形状
            try:
                # 确保数据类型匹配 - 转回与x_reduced相同的类型
                fused_vectors = fused_vectors.to(x_reduced.dtype)
                fused_features = fused_vectors.view(B, -1, x_reduced.size(-1))
                fused_features = self.norm_fused(fused_features)
            except Exception as e:
                swin_logger.warning(f"特征重塑或规范化失败: {e}")

                return x_reduced

            # 使用可学习的融合权重，但有上限
            alpha = torch.clamp(self.fusion_weight, 0.0, 0.5)
            final_output = (1 - alpha) * x_reduced + alpha * fused_features


            return final_output

        except Exception as e:
            swin_logger.warning(f"特征银行操作失败: {e}")

            return x_reduced

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"
