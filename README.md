# ProDyS: Self-Supervised Feature Learning with Prototype-Guided Context Fusion and Progressive Optimization

ProDyS is a self-supervised medical image segmentation framework designed to improve small-lesion recognition, boundary reconstruction, and feature generalization across data distributions.

- [Project documentation](docs/ProDyS_paper.md)
- [Method and qualitative figures](assets/figures/)
- [Conda environment](environment.yml)
- [Python requirements](requirements.txt)

## Main Content

Medical image segmentation requires both accurate localization and robust feature representation. ProDyS combines progressive dynamic optimization with prototype-guided self-supervision. The framework introduces Parallel Multi-Branch Enhancement (PMBE) for multi-scale detail representation, Soft-Hard mask-guided Dynamic Up-sampling (SHD-UP) for boundary-aware feature recovery, and Prototype-Clustering-driven Self-supervised feature Alignment (PCSA) for feature-space alignment across data distributions.

![Overall architecture of ProDyS](assets/figures/overall_architecture.png)

## Contributions

1. **PMBE** uses heterogeneous parallel branches and gradient-aware multi-scale processing to strengthen representations of small targets, subtle structures, and low-contrast regions.
2. **SHD-UP** uses multi-level mask guidance and progressive feature fusion to improve boundary continuity and structural consistency during up-sampling.
3. **PCSA** maintains a dynamic feature bank with prototype clustering and momentum updates to improve representation alignment and cross-domain generalization.

### Module overview

| Module | Role | Figure |
|---|---|---|
| PMBE | Multi-scale gradient-parallel feature enhancement | [PMBE](assets/figures/pmbe.png) |
| SHD-UP | Soft-hard mask-guided dynamic up-sampling | [SHD-UP](assets/figures/shd_up.png) |
| PCSA | Prototype-clustering-driven self-supervised feature alignment | [Feature bank](assets/figures/pcsa_feature_bank.png) |

## Results

| Dataset | DSC (%) | mIoU (%) |
|---|---:|---:|
| BUSI | **81.63** | - |
| ISIC2018 | **90.62** | **83.92** |
| CVC_ClinicDB | **90.82** | **83.92** |
| SMAE | **80.92** | **69.32** |

### Qualitative comparisons

| Dataset | Figure |
|---|---|
| BUSI | [Qualitative comparison](assets/figures/busi_comparison.png) |
| ISIC2018 | [Qualitative comparison](assets/figures/isic2018_comparison.png) |
| CVC_ClinicDB | [Qualitative comparison](assets/figures/cvc_comparison.png) |
| SMAE | [Qualitative comparison](assets/figures/smae_comparison.png) |
| SMAE / ISIC2018 | [Ablation figure](assets/figures/ablation.png) |

## Installation

Run the commands from the repository root. Training and evaluation require an NVIDIA GPU and a CUDA-enabled PyTorch installation. Create the Python environment:

```bash
conda env create -f environment.yml
conda activate prodys
```

Install PyTorch and torchvision for your CUDA driver using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the source imports:

```bash
pip install -r requirements.txt
```

The training entry point loads Swin-Tiny initialization from `./pretrained_ckpt/swin_tiny_patch4_window7_224.pth`, as set in [the configuration](configs/swin_tiny_patch4_window7_224_lite.yaml). Obtain the compatible Swin-Tiny weights using the [SCUNet++ setup instructions](https://github.com/JustlfC03/SCUNet-plusplus#1-download-pretrained-model), and place the file there, or change `MODEL.PRETRAIN_CKPT` in the YAML file to your local checkpoint. The checkpoint should contain a `model` state dictionary in the format expected by `net.load_from`.

## Data Preparation

Each split has a text file with one sample stem per line, without the extension. The `--list_dir` option points to the directory containing `train.txt` and `test.txt`. For BUSI and CVC_ClinicDB, use the following layout with `.png` images and masks:

```text
data/
└── BUSI/
    ├── lists/
    │   ├── train.txt
    │   └── test.txt
    ├── train/
    │   ├── images/case_001.png
    │   └── masks/case_001.png
    └── test/
        ├── images/case_002.png
        └── masks/case_002.png
```

Replace `BUSI` with `CVC_ClinicDB` for the CVC loader. For ISIC2018, keep the same directories but use `.jpg` images and `.png` or `.jpg` masks with matching stems. The loaders binarize masks at intensity 127 and resize training pairs to the configured input size; keep image/mask names synchronized. Use your own fixed train/test split files when comparing experiments.

The `Synapse` loader used for NPZ-format data expects a different layout:

```text
data/SMAE/
├── lists/
│   ├── train.txt
│   └── test.txt
├── train_npz/case_001.npz
└── test_vol_h5/case_002.npz
```

Each `.npz` must contain `image` and `label` arrays. In this loader, `image` is expected to be a two-dimensional grayscale image and `label` a matching mask encoded as 0/255; the loader converts the image to three channels and binarizes the mask. Despite the `test_vol_h5` directory name, this implementation reads `.npz` files there. Select this loader with `--dataset Synapse` and point `--root_path`/`--volume_path` to `data/SMAE`. Public datasets should be obtained from their original providers and used under their respective terms.

## Training and Evaluation

The following BUSI example enables all three method components: `--pbb` (multi-branch enhancement), `--x4` (enhanced up-sampling), and `--fb` (feature bank). The same entry points accept `--dataset isic2018` or `--dataset cvc` with the corresponding paths and file conventions above.

```bash
python train.py \
  --dataset busi \
  --root_path ./data/BUSI \
  --list_dir ./data/BUSI/lists \
  --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml \
  --output_dir ./output/busi \
  --device 0 --img_size 224 --batch_size 24 --max_epochs 200 \
  --fb --x4 --pbb
```

The training script uses SGD with momentum `0.9`, weight decay `1e-4`, and a polynomial learning-rate schedule; its default initial learning rate is `0.005`. It saves checkpoints under `--output_dir`, including `epoch_199.pth` for the 200-epoch example. Evaluate that checkpoint with the same module flags:

```bash
python test.py \
  --dataset busi \
  --volume_path ./data/BUSI \
  --list_dir ./data/BUSI/lists \
  --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml \
  --output_dir ./output/busi \
  --output_file_name ./output/busi/epoch_199.pth \
  --device 0 --img_size 224 \
  --fb --x4 --pbb
```

Evaluation prints Dice, HD95, sensitivity, specificity, accuracy, and IoU, and writes predicted and reference masks into `pred_image/` and `label_image/` in the current working directory.

## Repository Layout

| Path | Purpose |
|---|---|
| `train.py`, `trainer.py`, `test.py` | Training and evaluation entry points |
| `networks/` | Swin-Unet backbone and ProDyS modules |
| `datasets/` | Dataset readers and augmentation |
| `configs/` and `config.py` | Model configuration |
| `docs/` and `assets/figures/` | Method details and figures |

## Acknowledgements

The implementation builds on [SCUNet++](https://github.com/JustlfC03/SCUNet-plusplus) and the Swin-Unet architecture. The upstream MIT license notice is retained in [LICENSE](LICENSE).
