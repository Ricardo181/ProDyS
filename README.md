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

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate prodys
```

For a machine-specific CUDA installation, select the matching PyTorch command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining packages with:

```bash
pip install -r requirements.txt
```

The experimental configuration uses Python 3.10, 224x224 input images, batch size 8, 200 epochs, SGD, an initial learning rate of `1e-4`, momentum `0.9`, weight decay `1e-4`, and a minimum learning rate of `1e-5`.

## Data Preparation

The experiments use BUSI, ISIC2018, CVC_ClinicDB, and SMAE. Organize each dataset with a matching image/mask layout:

```text
data/
├── BUSI/
│   ├── images/
│   └── masks/
├── ISIC2018/
│   ├── images/
│   └── masks/
├── CVC_ClinicDB/
│   ├── images/
│   └── masks/
└── SMAE/
    ├── images/
    └── masks/
```

Dataset information used in the experiments:

| Dataset | Description | Scale reported in the experiments |
|---|---|---:|
| BUSI | Breast ultrasound lesion segmentation | - |
| ISIC2018 | Skin lesion segmentation | 2,694 images; 1,886 training and 808 testing images |
| CVC_ClinicDB | Colon polyp segmentation | 612 images at 384x288 pixels |
| SMAE | Superior mesenteric artery embolism segmentation | 626 images at approximately 512x512 pixels |

Prepare the data with the following conventions:

1. Keep each image and its binary mask under the corresponding `images/` and `masks/` directories.
2. Use the same filename stem for an image and its mask, for example `case_001.png` and `case_001.png`.
3. Resize input images to `224x224` for model input. Use nearest-neighbor interpolation for masks.
4. Apply random cropping and rotation to the training images and masks with the same random parameters.
5. Use the 7:3 training/testing split reported for ISIC2018; keep the split fixed when comparing models.

Public datasets should be obtained from their original dataset pages and used according to their respective terms. The SMAE directory follows the same layout.

## Acknowledgements

The project builds on the medical image segmentation literature and compares against U-Net, Transformer, Mamba, and hybrid segmentation models.

## Citation

```bibtex
@article{prodys,
  title  = {ProDyS: Self-Supervised Feature Learning with Prototype-Guided Context Fusion and Progressive Optimization},
  author = {ProDyS}
}
```
