#!/bin/bash

## Train
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/simplified_nerf_8v/330_blender_chair_8views_simple.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/simplified_nerf_8v/331_blender_drums_8views_simple.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/simplified_nerf_8v/332_blender_ficus_8views_simple.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/simplified_nerf_8v/334_blender_mic_8views_simple.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/simplified_nerf_8v/335_blender_ship_8views_simple.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/simplified_nerf_8v/312_blender_lego_8views_tune.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/simplified_nerf_8v/345_blender_hotdog_8views_simplest.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config configs/simplified_nerf_8v/347_blender_materials_8views_simpler.txt &
wait;

## Test
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/simplified_nerf_8v/345_blender_hotdog_8views_simplest.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/simplified_nerf_8v/347_blender_materials_8views_simpler.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/simplified_nerf_8v/335_blender_ship_8views_simple.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/simplified_nerf_8v/331_blender_drums_8views_simple.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/simplified_nerf_8v/330_blender_chair_8views_simple.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/simplified_nerf_8v/334_blender_mic_8views_simple.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --render_only --render_test --config configs/simplified_nerf_8v/332_blender_ficus_8views_simple.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --render_only --render_test --config configs/simplified_nerf_8v/312_blender_lego_8views_tune.txt &

## FID and KID
mkdir logs/nerf_simple_images_8
cp -r logs/330_blender_chair_8views_simple/testset_200000 logs/nerf_simple_images_8/chair_testset_200000
cp -r logs/334_blender_mic_8views_simple/testset_200000 logs/nerf_simple_images_8/mic_testset_200000
cp -r logs/332_blender_ficus_8views_simple/testset_200000 logs/nerf_simple_images_8/ficus_testset_200000
cp -r logs/312_blender_lego_8views_tune/testset_200000 logs/nerf_simple_images_8/lego_testset_200000
cp -r logs/345_blender_hotdog_8views_simplest/testset_200000 logs/nerf_simple_images_8/hotdog_testset_200000
cp -r logs/347_blender_materials_8views_simpler/testset_200000 logs/nerf_simple_images_8/materials_testset_200000
cp -r logs/335_blender_ship_8views_simple/testset_200000 logs/nerf_simple_images_8/ship_testset_200000
cp -r logs/331_blender_drums_8views_simple/testset_200000 logs/nerf_simple_images_8/drums_testset_200000

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_simple_images_8/ data/nerf_synthetic_400