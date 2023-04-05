# Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis
[Website](https://ajayj.com/dietnerf)  |  [ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/html/Jain_Putting_NeRF_on_a_Diet_Semantically_Consistent_Few-Shot_View_Synthesis_ICCV_2021_paper.html)  |  [arXiv](https://arxiv.org/abs/2104.00677)  | [Twitter](https://twitter.com/ajayj_/status/1379475290154356738)

![Diagram overviewing DietNeRF's training procedure](https://d33wubrfki0l68.cloudfront.net/360965d431284b958d86082f9b49d53a0356b632/85d1a/dietnerf/assets/img/dietnerf_method_anim_50p.gif)

This repository contains the official implementation of DietNeRF, a system that reconstructs 3D scenes from a few posed photos.

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

## Experiments on the DTU dataset
Coming soon. Our paper also fine-tunes pixelNeRF on DTU scenes for 1-shot view synthesis.

## Citation and acknowledgements
If DietNeRF is relevant to your project, please cite our associated [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Jain_Putting_NeRF_on_a_Diet_Semantically_Consistent_Few-Shot_View_Synthesis_ICCV_2021_paper.html):
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

