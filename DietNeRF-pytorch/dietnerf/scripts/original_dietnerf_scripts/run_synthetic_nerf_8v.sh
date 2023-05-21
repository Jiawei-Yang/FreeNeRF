#!/bin/bash

## Train
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/nerf_8v/230_blender_chair_8views.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/nerf_8v/231_blender_drums_8views.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/nerf_8v/232_blender_ficus_8views.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/nerf_8v/233_blender_lego_8views.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/nerf_8v/234_blender_mic_8views.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/nerf_8v/235_blender_ship_8views.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/nerf_8v/236_blender_hotdog_8views.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config configs/nerf_8v/237_blender_materials_8views.txt &
wait;

## Test
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/nerf_8v/230_blender_chair_8views.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/nerf_8v/231_blender_drums_8views.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/nerf_8v/232_blender_ficus_8views.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/nerf_8v/233_blender_lego_8views.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/nerf_8v/234_blender_mic_8views.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/nerf_8v/235_blender_ship_8views.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --render_only --render_test --config configs/nerf_8v/236_blender_hotdog_8views.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --render_only --render_test --config configs/nerf_8v/237_blender_materials_8views.txt &
wait;

## Evaluate FID and KID
mkdir logs/nerf_images_8
cp -r logs/230_blender_chair_8views/testset_200000/ logs/nerf_images_8/chair_testset_200000
cp -r logs/231_blender_drums_8views/testset_200000/ logs/nerf_images_8/drums_testset_200000
cp -r logs/232_blender_ficus_8views/testset_200000/ logs/nerf_images_8/ficus_testset_200000
cp -r logs/233_blender_lego_8views/testset_200000/ logs/nerf_images_8/lego_testset_200000
cp -r logs/235_blender_ship_8views/testset_200000/ logs/nerf_images_8/ship_testset_200000
cp -r logs/234_blender_mic_8views/testset_200000/ logs/nerf_images_8/mic_testset_200000
cp -r logs/236_blender_hotdog_8views/testset_200000/ logs/nerf_images_8/hotdog_testset_200000
cp -r logs/237_blender_materials_8views/testset_200000/ logs/nerf_images_8/materials_testset_200000

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_8/ data/nerf_synthetic_400