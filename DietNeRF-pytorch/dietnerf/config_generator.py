from collections import namedtuple

import fire

Scene = namedtuple('Scene', 'name datadir')

SYNTHETIC_SCENES = [
    Scene('chair', './data/nerf_synthetic/chair'),
    Scene('drums', './data/nerf_synthetic/drums'),
    Scene('ficus', './data/nerf_synthetic/ficus'),
    Scene('lego', './data/nerf_synthetic/lego'),
    Scene('mic', './data/nerf_synthetic/mic'),
    Scene('ship', './data/nerf_synthetic/ship'),
]

base_synthetic_config = \
"""basedir = ./logs
dataset_type = blender
no_batching = True
use_viewdirs = True
white_bkgd = True
lrate_decay = 500
N_samples = 64
N_importance = 128
N_rand = 1024
precrop_iters = 500
precrop_frac = 0.5
half_res = True
"""

def make_synthetic_scenes(start_id, max_train_views=-1, consistency_loss=False):
    commands = []
    for i, scene in enumerate(SYNTHETIC_SCENES):
        expname = f"{start_id + i}_blender_{scene.name}_{max_train_views}views"
        if consistency_loss:
            expname += "_ctr"

        config = \
f"""expname = {expname}
datadir = {scene.datadir}
{base_synthetic_config}
## Additional arguments
max_train_views = {max_train_views}
i_log_raw_hist = 50
i_video = 6250
save_splits = True
checkpoint_rendering = True"""

        if consistency_loss:
            config = \
f"""{config}
## Computational options relevant for rendering
pixel_interp_mode = bilinear
feature_interp_mode = bilinear
checkpoint_rendering = True
i_log_ctr_img = 10

## Shared rendering loss options
render_loss_interval = 10
render_nH = 168
render_nW = 168
render_jitter_rays = True
render_poses = uniform
render_theta_range = [-180, 180]
render_phi_range = [-90, 0]
render_radius_range = [3.5, 4.5]

## Consistency loss options
consistency_loss = consistent_with_target_rep
consistency_loss_lam = 0.1
consistency_loss_lam0 = 0.1
consistency_model_type = clip_vit
consistency_size = 224
consistency_loss_comparison = cosine_sim"""

        out_path = f'configs/{expname}.txt'
        print("==== WRITING TO", out_path)
        print(config)
        with open(out_path, 'w') as f:
            f.write(config)
        print("=============================")

        command = f"CUDA_VISIBLE_DEVICES={i} python run_nerf.py --config {out_path} &"
        commands.append(command)

    print("=========== COMMANDS")
    commands = '#!/bin/bash\n' + '\n'.join(commands)
    print(commands)
    # with open(f'scripts/{start_id}_run_synthetic_{max_train_views}views.sh')



if __name__=='__main__':
    fire.Fire()
