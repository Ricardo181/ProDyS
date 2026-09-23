import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_busi import BUSI_dataset
from datasets.dataset_isic import ISIC_dataset
from datasets.dataset_thyroid import Thyroid_dataset
from datasets.dataset_ph2 import PH2_dataset
from datasets.dataset_montgomery import Montgomery_dataset
from datasets.dataset_promise import Promise_dataset
from datasets.dataset_physionet import Physionet_dataset
from datasets.dataset_cvc import CVC_dataset
from datasets.dataset_kvasir_seg import Kvasir_Seg_dataset
from datasets.dataset_jsrt import Jsrt_dataset

from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer
from config import get_config
from torchvision import transforms

# 创建专属于test模块的logger
logger = logging.getLogger('test')
"""
--dataset Synapse
--cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml
--is_saveni
--volume_path ./datasets/Synapse
--output_dir ./output
--max_epoch 150
--base_lr 0.05
--img_size 224
--batch_size 1
"""

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./datasets/Synapse/',
                    help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
# parser.add_argument('--num_classes', type=int,
#                     default=9, help='output channel of network')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', default='./output', type=str, help='output dir')
parser.add_argument('--output_file_name', default=None, type=str, help='output file name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--device', type=int, required=True,  help='cuda', )

parser.add_argument('--x4', action='store_true')
parser.add_argument('--pbb', action='store_true')
parser.add_argument('--fb', action='store_true')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
    # print(args.volume_path)
config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logger.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    # 初始化 metric_list 以存储 6 个指标 (Dice, HD95, SE, SP, ACC, IoU) for num_classes - 1 个类别
    metric_list = np.zeros((args.num_classes - 1, 6))

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # --- Preprocessing Verification ---
        if i_batch == 0: # Print stats only for the first batch to avoid clutter
            logger.info(f"-- Verifying Preprocessing for Batch {i_batch} ---")
            logger.info(f"  Image Tensor Shape: {image.shape}")
            # Ensure image is float for stats calculation
            image_float = image.float()
            logger.info(f"  Image Tensor dtype: {image_float.dtype}")
            logger.info(f"  Min value: {torch.min(image_float):.6f}")
            logger.info(f"  Max value: {torch.max(image_float):.6f}")
            logger.info(f"  Mean value: {torch.mean(image_float):.6f}")
            logger.info(f"  Std value: {torch.std(image_float):.6f}")
            logger.info(f"-----------------------------------------------")

        # test_single_volume 现在返回一个包含 (dice, hd95, se, sp, acc, iou) 元组的列表，每个元组对应一个类别
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, cuda=args.device)
        # 累积每个类别的指标
        metric_list += np.array(metric_i)

    # 计算所有测试样本的平均指标
    metric_list = metric_list / len(db_test)

    # 记录每个类别的平均指标
    for i in range(1, args.num_classes):
        class_metrics = metric_list[i - 1]
        logger.info('Mean class %d metrics:' % i)
        logger.info('  Dice: %f' % class_metrics[0])
        logger.info('  HD95: %f' % class_metrics[1])
        logger.info('  SE:   %f' % class_metrics[2])
        logger.info('  SP:   %f' % class_metrics[3])
        logger.info('  ACC:  %f' % class_metrics[4])
        logger.info('  IoU:  %f' % class_metrics[5])

    # 计算并记录所有类别的平均指标
    performance = np.mean(metric_list, axis=0)
    mean_dice = performance[0]
    mean_hd95 = performance[1]
    mean_se = performance[2]
    mean_sp = performance[3]
    mean_acc = performance[4]
    mean_iou = performance[5]

    logger.info('Testing performance in best val model:')
    logger.info('  Mean Dice: %f' % mean_dice)
    logger.info('  Mean HD95: %f' % mean_hd95)
    logger.info('  Mean SE:   %f' % mean_se)
    logger.info('  Mean SP:   %f' % mean_sp)
    logger.info('  Mean ACC:  %f' % mean_acc)
    logger.info('  Mean IoU:  %f' % mean_iou)

    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            # 'num_classes': 9,
            'num_classes': 2,
            'z_spacing': 1,
        },
        'busi': {
            'Dataset': BUSI_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 2,
            'z_spacing': 1,
        },
        'isic2017': {
            'Dataset': ISIC_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 2,
            'z_spacing': 1,
        },
        'isic2018': {
            'Dataset': ISIC_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 2,
            'z_spacing': 1,
        },
        'ph2': {
            'Dataset': PH2_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 2,
            'z_spacing': 1,
        },
        'thyroid': {
            'Dataset': Thyroid_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': 2,
            'z_spacing': 1,
        },
        'montgomery': {
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'Dataset': Montgomery_dataset,
            'num_classes': 2,
            'z_spacing': 1,
        },
        'promise': {
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'Dataset': Promise_dataset,
            'num_classes': 2,
            'z_spacing': 1,
        },
         'physionet': {
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'Dataset': Physionet_dataset,
            'num_classes': 2,
             'z_spacing': 1,
        },
        'cvc': {
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'Dataset': CVC_dataset,
            'num_classes': 2,
             'z_spacing': 1,
        },'kvasir': {
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'Dataset': Kvasir_Seg_dataset,
            'num_classes': 2,
             'z_spacing': 1,
        },
        'jsrt': {
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'Dataset': Jsrt_dataset,
            'num_classes': 2,
             'z_spacing': 1,
        },

    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes,use_feature_bank=args.fb, use_enhanced_up_x4=args.x4, use_pbb=args.pbb).cuda(args.device)
    print(net)
    if args.output_file_name is not None:
        snapshot = args.output_file_name
    else:
        snapshot = os.path.join(args.output_dir, 'best_model.pth')
        if not os.path.exists(snapshot):
            snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
            # snapshot = snapshot.replace('best_model', 'epoch_' + str(199))

    # Construct the target device string (e.g., 'cuda:1')
    target_device = f'cuda:{args.device}'
    # Load state dict mapping directly to the target GPU device
    print(f"Loading state dict from {snapshot} directly to {target_device}")
    state_dict = torch.load(snapshot, map_location=target_device, weights_only=True)
    msg = net.load_state_dict(state_dict, strict=False)
    # Previous CPU mapping lines removed
    # state_dict = torch.load(snapshot, map_location='cpu', weights_only=True)
    # msg = net.load_state_dict(state_dict, strict=False)
    # Original line commented out for reference:
    # msg = net.load_state_dict(torch.load(snapshot,weights_only=True),strict=False,)
    print("self trained swin unet", msg)
    print(args.output_dir)
    snapshot_name = args.output_dir.split('/')[-1]
    print(snapshot_name)

    # 清除已有的处理器，避免重复添加
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 配置test专属的logger
    logger.setLevel(logging.INFO)

    # 添加控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console_handler)

    # 禁用根日志处理器的传播，避免日志重复
    logger.propagate = False

    logger.info(str(args))
    logger.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
