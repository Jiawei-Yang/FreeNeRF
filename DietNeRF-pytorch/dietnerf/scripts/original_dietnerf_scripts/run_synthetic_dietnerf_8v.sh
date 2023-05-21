#!/bin/bash

## Train
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/dietnerf_8v/250_blender_chair_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/dietnerf_8v/251_blender_drums_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/dietnerf_8v/252_blender_ficus_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/dietnerf_8v/253_blender_lego_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/dietnerf_8v/254_blender_mic_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/dietnerf_8v/255_blender_ship_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/dietnerf_8v/256_blender_hotdog_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config configs/dietnerf_8v/257_blender_materials_8views_ctr.txt &
wait;

## Test with 8 views
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/dietnerf_8v/250_blender_chair_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/dietnerf_8v/251_blender_drums_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/dietnerf_8v/252_blender_ficus_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/dietnerf_8v/253_blender_lego_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/dietnerf_8v/254_blender_mic_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/dietnerf_8v/255_blender_ship_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --render_only --render_test --config configs/dietnerf_8v/256_blender_hotdog_8views_ctr.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --render_only --render_test --config configs/dietnerf_8v/257_blender_materials_8views_ctr.txt &
wait;

## Evaluate FID and KID
mkdir logs/dietnerf_images_8
cp -r logs/250_blender_chair_8views_ctr/testset_200000 logs/dietnerf_images_8/chair_testset_200000
cp -r logs/251_blender_drums_8views_ctr/testset_200000 logs/dietnerf_images_8/drums_testset_200000
cp -r logs/252_blender_ficus_8views_ctr/testset_200000 logs/dietnerf_images_8/ficus_testset_200000
cp -r logs/253_blender_lego_8views_ctr/testset_200000 logs/dietnerf_images_8/lego_testset_200000
cp -r logs/254_blender_mic_8views_ctr/testset_200000 logs/dietnerf_images_8/mic_testset_200000
cp -r logs/255_blender_ship_8views_ctr/testset_200000 logs/dietnerf_images_8/ship_testset_200000
cp -r logs/256_blender_hotdog_8views_ctr/testset_200000 logs/dietnerf_images_8/hotdog_testset_200000
cp -r logs/257_blender_materials_8views_ctr/testset_200000 logs/dietnerf_images_8/materials_testset_200000

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/dietnerf_images_8/ data/nerf_synthetic_400
