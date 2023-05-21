#!/bin/bash
## Train
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/dietnerf_ft_8v/282_blender_mic_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/dietnerf_ft_8v/283_blender_chair_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/dietnerf_ft_8v/284_blender_drums_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/dietnerf_ft_8v/285_blender_ficus_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/dietnerf_ft_8v/286_blender_lego_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/dietnerf_ft_8v/289_blender_ship_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/dietnerf_ft_8v/287_blender_materials_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config configs/dietnerf_ft_8v/288_blender_hotdog_8views_ftctr.txt &
wait;

## Test with 8 views at 250k iterations total
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/dietnerf_ft_8v/283_blender_chair_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/dietnerf_ft_8v/284_blender_drums_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/dietnerf_ft_8v/285_blender_ficus_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/dietnerf_ft_8v/286_blender_lego_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/dietnerf_ft_8v/282_blender_mic_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/dietnerf_ft_8v/289_blender_ship_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/dietnerf_ft_8v/288_blender_hotdog_8views_ftctr.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --render_only --render_test --reload_iter 250000 --config configs/dietnerf_ft_8v/287_blender_materials_8views_ftctr.txt &
wait;

## Evaluate FID and KID
mkdir logs/dietnerfft_images_8
cp -r logs/282_blender_mic_8views_ftctr254/testset_250000/ logs/dietnerfft_images_8/mic_testset_250000
cp -r logs/283_blender_chair_8views_ftctr250/testset_250000/ logs/dietnerfft_images_8/chair_testset_250000
cp -r logs/284_blender_drums_8views_ftctr251/testset_250000/ logs/dietnerfft_images_8/drums_testset_250000
cp -r logs/285_blender_ficus_8views_ftctr252/testset_250000/ logs/dietnerfft_images_8/ficus_testset_250000
cp -r logs/286_blender_lego_8views_ftctr253/testset_250000/ logs/dietnerfft_images_8/lego_testset_250000
cp -r logs/287_blender_materials_8views_ftctr257/testset_250000/ logs/dietnerfft_images_8/materials_testset_250000
cp -r logs/288_blender_hotdog_8views_ftctr256/testset_250000/ logs/dietnerfft_images_8/hotdog_testset_250000
cp -r logs/289_blender_ship_8views_ftctr255/testset_250000/ logs/dietnerfft_images_8/ship_testset_250000

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/dietnerfft_images_8/ data/nerf_synthetic_400