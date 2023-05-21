#!/bin/bash

## Train
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --config configs/unseen_side_14v/314_blender_lego_right_dietnerf.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/unseen_side_14v/315_blender_lego_nerf_right.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --config configs/unseen_side_14v/317_blender_lego_right_nerf_simple.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/unseen_side_14v/319_blender_lego_right_dietnerf_simple.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --config configs/unseen_side_14v/321_blender_lego_right_ftctr314_130k.txt &
# Skip training NeRF with 100 views; assume this has already been done. (configs/nerf_100v/lego.txt)
wait;

## Test
CUDA_VISIBLE_DEVICES=0 python run_nerf.py --render_only --render_test --config configs/unseen_side_14v/314_blender_lego_right_dietnerf.txt &
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --render_only --render_test --config configs/unseen_side_14v/315_blender_lego_nerf_right.txt &
CUDA_VISIBLE_DEVICES=2 python run_nerf.py --render_only --render_test --config configs/unseen_side_14v/317_blender_lego_right_nerf_simple.txt &
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --render_only --render_test --config configs/unseen_side_14v/319_blender_lego_right_dietnerf_simple.txt &
CUDA_VISIBLE_DEVICES=4 python run_nerf.py --render_only --render_test --config configs/unseen_side_14v/321_blender_lego_right_ftctr314_130k.txt &
CUDA_VISIBLE_DEVICES=5 python run_nerf.py --render_only --render_test --config configs/nerf_100v/lego.txt &
wait;