# [CVPR23] FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization

## [Project Page](https://jiawei-yang.github.io/FreeNeRF/) | [Paper](https://arxiv.org/abs/2303.07418)

This repository contains the code release for the CVPR 2023 project
> [**FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization**](https://arxiv.org/abs/2303.07418),  
> Jiawei Yang, Marco Pavone, and Yue Wang   
> Computer Vision and Pattern Recognition (CVPR), 2023


This code is based on [DietNeRF implementation](https://github.com/ajayjain/DietNeRF). We provide some setup instructions below. For more details, please refer to the [DietNeRF README](https://github.com/ajayjain/DietNeRF/blob/master/README.md) for more details.
## Setup

We use the following folder structure:
```
dietnerf/
  logs/ (images, videos, checkpoints)
  data/
    nerf_synthetic/
  configs/ (run configuration files)
CLIP/ (Fork of OpenAI's clip repository with a wrapper)
```

Create conda environment:
```
conda create -n dietnerf python=3.9
conda activate dietnerf
```

Set up requirements and our fork of CLIP:
```
pip install -r requirements.txt
cd CLIP
pip install -e .
```

Login to Weights & Biases:
```
wandb login
```

## Experiments on the Realistic Synthetic dataset
Realistic Synthetic experiments are implemented in the `./dietnerf` subdirectory.

You need to download datasets
from [NeRF's Google Drive folder](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
The dataset was used in the original NeRF paper by Mildenhall et al. For example,
```
mkdir dietnerf/logs/ dietnerf/data/
cd dietnerf/data
pip install gdown
gdown --id 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG -O nerf_synthetic.zip
unzip nerf_synthetic.zip
rm -r __MACOSX
```

Then, shrink images to 400x400:
```
python dietnerf/scripts/bulk_shrink_images.py "dietnerf/data/nerf_synthetic/*/*/*.png" dietnerf/data/nerf_synthetic_400_rgb/ True
```
These images are used for FID/KID computation. The `dietnerf/run_nerf.py` training and evaluation code automatically shrinks images with the `--half_res` argument.

Each experiment has a config file stored in `dietnerf/configs/`. Scripts in `dietnerf/scripts/` can be run to train and evaluate models.


### Sample scripts from DietNeRF
Below are sample commands to run experiments from DietNeRF:

Run these scripts from `./dietnerf`.
The scripts assume you are running one script at a time on a server with 8 NVIDIA GPUs.
```
cd dietnerf
export WANDB_ENTITY=<your wandb username>

# NeRF baselines
sh scripts/run_synthetic_nerf_100v.sh
sh scripts/run_synthetic_nerf_8v.sh
sh scripts/run_synthetic_simplified_nerf_8v.sh

# DietNeRF with 8 observed views
sh scripts/run_synthetic_dietnerf_8v.sh
sh scripts/run_synthetic_dietnerf_ft_8v.sh

# NeRF and DietNeRF with partial observability
sh scripts/run_synthetic_unseen_side_14v.sh
```

## Sample scripts for FreeNeRF
```
# train on a single GPU
CUDA_VISIBLE_DEVICES=0 python run_nerf.py \
    --config configs/freenerf_8v/freenerf_8v_50k_base05.txt \
    --datadir data/nerf_synthetic/chair \
    --expname chair_freenerf_reg0.5 

# test on a single GPU
CUDA_VISIBLE_DEVICES=0 python run_nerf.py \
    --render_only \
    --render_test \
    --config configs/freenerf_8v/freenerf_8v_50k_base05.txt \
    --datadir data/nerf_synthetic/chair \
    --expname chair_freenerf_reg0.5
```
To change to other samples, change the `--datadir` and `--expname` accordingly, e.g., `--datadir data/nerf_synthetic/lego --expname lego_freenerf_reg0.5`.

We found that training nerf for 50k iterations or less is sufficient for FreeNeRF to achieve good performance. Training 200k iterations will lead to similar performance but takes much longer time.

## Citation and acknowledgements



If you find our work useful, consider citing:
```
@InProceedings{Yang2023FreeNeRF,
    author    = {Jiawei Yang and Marco Pavone and Yue Wang},},  
    title     = {FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization},
    booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023},
}
```

```
@InProceedings{Jain_2021_ICCV,
    author    = {Jain, Ajay and Tancik, Matthew and Abbeel, Pieter},
    title     = {Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {5885-5894}
}
```
This code is based on Yen-Chen Lin's [PyTorch implementation of NeRF](https://github.com/yenchenlin/nerf-pytorch) and the [official pixelNeRF code](https://github.com/sxyu/pixel-nerf).

