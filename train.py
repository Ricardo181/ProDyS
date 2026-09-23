import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer
from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_busi import BUSI_dataset
from datasets.dataset_isic import ISIC_dataset
from datasets.dataset_thyroid import Thyroid_dataset
from datasets.dataset_montgomery import Montgomery_dataset
from datasets.dataset_promise import Promise_dataset
from datasets.dataset_physionet import Physionet_dataset
from datasets.dataset_cvc import CVC_dataset
from datasets.dataset_ph2 import PH2_dataset
from datasets.dataset_jsrt import Jsrt_dataset
from datasets.dataset_kvasir_seg import Kvasir_Seg_dataset
"""
--dataset Synapse
--cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml
--root_path  ./datasets/Synapse
--max_epochs 1500
--output_dir ./output
--img_size 224
--base_lr 0.005
--batch_size 24
"""

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./datasets/Synapse/', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
# parser.add_argument('--num_classes', type=int,
#                     default=9, help='output channel of network')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', default='./output', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', default="./configs/swin_tiny_patch4_window7_224_lite.yaml" )
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

parser.add_argument('--device', type=int, default=0, help='cuda')

parser.add_argument('--x4',  action='store_true')
parser.add_argument('--pbb',  action='store_true')
parser.add_argument('--fb',  action='store_true')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)

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
    torch.cuda.set_device(args.device)
    # 启用 TF32 以获得更好的性能（需要兼容硬件和 PyTorch >= 1.7）
    torch.set_float32_matmul_precision('high')
    torch.cuda.manual_seed(args.seed)
    torch.cuda.empty_cache()
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': Synapse_dataset,
            # 'num_classes': 9,
            'num_classes': 2,
        },
        'busi': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': BUSI_dataset,
            'num_classes': 2,
        },
        'isic2017': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': ISIC_dataset,
            'num_classes': 2,
        },
        'isic2018': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': ISIC_dataset,
            'num_classes': 2,
        },
        'thyroid': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': Thyroid_dataset,
            'num_classes': 2,
        },
        'montgomery': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': Montgomery_dataset,
            'num_classes': 2,
        },
        'promise': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': Promise_dataset,
            'num_classes': 2,
        },
         'physionet': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': Physionet_dataset,
            'num_classes': 2,
        },
         'cvc': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': CVC_dataset,
            'num_classes': 2,
        },
        'ph2': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': PH2_dataset,
            'num_classes': 2,
        },
        'jsrt': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': Jsrt_dataset,
            'num_classes': 2,
        },
        'kvasir': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'Dataset': Kvasir_Seg_dataset,
            'num_classes': 2,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']

    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes,use_feature_bank=args.fb, use_enhanced_up_x4=args.x4, use_pbb=args.pbb).cuda(args.device)

    net.load_from(config)

    trainer(args, net, args.output_dir)
