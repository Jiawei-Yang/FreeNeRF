#!/bin/bash

## Train
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/nerf_100v/chair.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/nerf_100v/drums.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/nerf_100v/ficus.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/nerf_100v/lego.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/nerf_100v/mic.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --config configs/nerf_100v/ship.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --config configs/nerf_100v/hotdog.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --config configs/nerf_100v/materials.txt &
wait;

## Test
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/nerf_100v/lego.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/nerf_100v/mic.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/nerf_100v/ship.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/nerf_100v/chair.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/nerf_100v/drums.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/nerf_100v/ficus.txt &
CUDA_VISIBLE_DEVICES=6 python run_nerf.py --render_only --render_test --config configs/nerf_100v/hotdog.txt &
CUDA_VISIBLE_DEVICES=7 python run_nerf.py --render_only --render_test --config configs/nerf_100v/materials.txt &
wait;

## Evaluate FID and KID
mkdir logs/nerf_images_100
cp -r logs/blender_paper_lego/testset_200000/ logs/nerf_images_100/lego_testset_200000
cp -r logs/blender_paper_mic/testset_200000/ logs/nerf_images_100/mic_testset_200000
cp -r logs/blender_paper_ship/testset_200000/ logs/nerf_images_100/ship_testset_200000
cp -r logs/blender_paper_chair/testset_200000/ logs/nerf_images_100/chair_testset_200000
cp -r logs/blender_paper_drums/testset_200000/ logs/nerf_images_100/drums_testset_200000
cp -r logs/blender_paper_ficus/testset_200000/ logs/nerf_images_100/ficus_testset_200000
cp -r logs/blender_paper_hotdog/testset_200000/ logs/nerf_images_100/hotdog_testset_200000
cp -r logs/blender_paper_materials/testset_200000/ logs/nerf_images_100/materials_testset_200000

fidelity --gpu 0 --samples-find-deep --kid-subset-size 200 --isc --fid --kid logs/nerf_images_100/ data/nerf_synthetic_400