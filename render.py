# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render script for RegNeRF."""
import functools
from os import path
import time
import os

from absl import app
import flax
from flax.training import checkpoints
from internal import configs, math, datasets, models, utils  # pylint: disable=g-multiple-import
import jax
from jax import random
import jax.numpy as jnp
import glob 
import numpy as np
import mediapy as media
from matplotlib import cm

configs.define_common_flags()
jax.config.parse_flags_with_absl()



def create_videos(config, base_dir, out_dir, out_name, num_frames):
  """Creates videos out of the images saved to disk.
  Credits to: https://github.com/google-research/multinerf/blob/main/render.py
  """
  names = [n for n in config.checkpoint_dir.split('/') if n]
  # Last two parts of checkpoint path are experiment name and scene name.
  exp_name, scene_name = names[-2:]
  video_prefix = f'{scene_name}_{exp_name}_{out_name}'

  zpad = max(3, len(str(num_frames - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  # utils.makedirs(base_dir, exist_ok=True)
  depth_curve_fn = lambda x: -jnp.log(x + jnp.finfo(jnp.float32).eps)

  # Load one example frame to get image shape and depth range.
  depth_file = os.path.join(out_dir, f'distance_mean_{idx_to_str(0)}.tiff')
  depth_frame = utils.load_img(depth_file)
  shape = depth_frame.shape
  p = 0.5
  distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
  lo, hi = [depth_curve_fn(x) for x in distance_limits]
  print(f'Video shape is {shape[:2]}')

  video_kwargs = {
      'shape': shape[:2],
      'codec': 'h264',
      'fps': 60,
      'crf': 18,
  }

  for k in ['color', 'normals', 'acc', 'distance_mean', 'distance_median']:
    video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
    input_format = 'gray' if k == 'acc' else 'rgb'
    file_ext = 'png' if k in ['color', 'normals'] else 'tiff'
    idx = 0
    file0 = os.path.join(out_dir, f'{k}_{idx_to_str(0)}.{file_ext}')
    if not utils.file_exists(file0):
      print(f'Images missing for tag {k}')
      continue
    print(f'Making video {video_file}...')
    with media.VideoWriter(
        video_file, **video_kwargs, input_format=input_format) as writer:
      for idx in range(num_frames):
        img_file = os.path.join(out_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
        if not utils.file_exists(img_file):
          ValueError(f'Image file {img_file} does not exist.')
        img = utils.load_img(img_file)
        if k in ['color', 'normals']:
          img = img / 255.
        elif k.startswith('distance'):
          img = depth_curve_fn(img)
          img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
          img = cm.get_cmap('turbo')(img)[..., :3]

        frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
        writer.add_image(frame)
        idx += 1

def main(unused_argv):

  config = configs.load_config(save_config=False)
  config.render_path = True

  dataset = datasets.load_dataset('test', config.data_dir, config)
  model, init_variables = models.construct_mipnerf(
      random.PRNGKey(20200823),
      dataset.peek()['rays'],
      config)
  optimizer = flax.optim.Adam(config.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables

  # Pre-define depth ranges for more across-settings consistent visualizations
  if config.dataset_loader == 'llff':
    eval_dict = {'fern': [0.059100067913532256, 0.8538959634304046],
                 'flower': [0.2099738734960556, 0.996519325375557],
                 'fortress': [0.3405687987804413, 0.8795422136783599],
                 'horns': [0.3501826047897339, 0.9596474349498749],
                 'leaves': [0.00022197533398866584, 0.9934533953666687],
                 'orchids': [0.23377860009670257, 0.9828365403413772],
                 'room': [0.4059941208362579, 0.9502887094020843],
                 'trex': [0.016071857213974, 0.9458529788255692]}
    lo, hi = eval_dict[config.llff_scan]  # pylint: disable=unused-variable
  elif config.dataset_loader == 'dtu':
    eval_dict = {'scan8': [0.9593777, 1.5342957],
                 'scan21': [0.98255014, 1.7484968],
                 'scan30': [1.1381109, 1.6074754],
                 'scan31': [1.0627427, 1.6069319],
                 'scan34': [1.1172018, 1.5005568],
                 'scan38': [1.0385504, 1.5373354],
                 'scan40': [0.8312144, 1.62111],
                 'scan41': [0.9469194, 1.5374442],
                 'scan45': [1.0098513, 1.5830635],
                 'scan55': [0.85020584, 1.513227],
                 'scan63': [1.1894969, 1.7325872],
                 'scan82': [1.0984676, 1.7162027],
                 'scan103': [1.0771852, 1.5858444],
                 'scan110': [0.96143025, 1.5147997],
                 'scan114': [0.96940583, 1.548706]}
    lo, hi = eval_dict[config.dtu_scan]

  path_fn = lambda x: path.join(out_dir, x)

  # Fix for loading pre-trained models.
  try:
    state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
  except:  # pylint: disable=bare-except
    print('Using pre-trained model.')
    state_dict = checkpoints.restore_checkpoint(config.checkpoint_dir, None)
    for i in [9, 17]:
      del state_dict['optimizer']['target']['params']['MLP_0'][f'Dense_{i}']
    state_dict['optimizer']['target']['params']['MLP_0'][
        'Dense_9'] = state_dict['optimizer']['target']['params']['MLP_0'][
            'Dense_18']
    state_dict['optimizer']['target']['params']['MLP_0'][
        'Dense_10'] = state_dict['optimizer']['target']['params']['MLP_0'][
            'Dense_19']
    state_dict['optimizer']['target']['params']['MLP_0'][
        'Dense_11'] = state_dict['optimizer']['target']['params']['MLP_0'][
            'Dense_20']
    del state_dict['optimizerd']
    state = flax.serialization.from_state_dict(state, state_dict)

  step = int(state.optimizer.state.step)
  print(f'Rendering checkpoint at step {step}.')

  # --- FreeNeRF add-ons --- #
  if config.freq_reg:
    # Compute frequency regularization masks for the current step.
    freq_reg_mask = (
      math.get_freq_reg_mask(99, step, config.freq_reg_end, config.max_vis_freq_ratio),
      math.get_freq_reg_mask(27, step, config.freq_reg_end, config.max_vis_freq_ratio))
    def render_eval_fn(variables, _, rays):
      return jax.lax.all_gather(
          model.apply(
              variables,
              None,  # Deterministic.
              rays,
              resample_padding=config.resample_padding_final,
              compute_extras=True,
              freq_reg_mask=freq_reg_mask)[0], axis_name='batch')
  else:
    def render_eval_fn(variables, _, rays):
      return jax.lax.all_gather(
          model.apply(
              variables,
              None,  # Deterministic.
              rays,
              resample_padding=config.resample_padding_final,
              compute_extras=True)[0], axis_name='batch')
  # pmap over only the data input.
  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(None, None, 0),
      donate_argnums=2,
      axis_name='batch',
  )
  # --- FreeNeRF add-ons --- #
  
  out_name = 'path_renders' if config.render_path else 'test_preds'
  out_name = f'{out_name}_step_{step}'
  base_dir = config.render_dir
  if base_dir is None:
    base_dir = config.checkpoint_dir
  out_dir = path.join(base_dir, out_name)
  if not utils.isdir(out_dir):
    utils.makedirs(out_dir)

  for idx in range(dataset.size):
    print(f'Evaluating image {idx+1}/{dataset.size}')
    eval_start_time = time.time()
    batch = next(dataset)
    rendering = models.render_image(
        functools.partial(render_eval_pfn, state.optimizer.target),
        batch['rays'],
        None,
        config)
    print(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

    if jax.host_id() != 0:  # Only record via host 0.
      continue

    utils.save_img_u8(rendering['rgb'], path_fn(f'color_{idx:03d}.png'))
    # time.sleep(3)
    utils.save_img_u8(rendering['normals'] / 2. + 0.5,
                      path_fn(f'normals_{idx:03d}.png'))
    # time.sleep(3)
    utils.save_img_f32(rendering['distance_mean'],
                       path_fn(f'distance_mean_{idx:03d}.tiff'))
    # time.sleep(3)
    utils.save_img_f32(rendering['distance_median'],
                       path_fn(f'distance_median_{idx:03d}.tiff'))
    # time.sleep(3)
    utils.save_img_f32(rendering['acc'], path_fn(f'acc_{idx:03d}.tiff'))

  ## ---- Create videos ---- ##
  num_files = len(glob.glob(path_fn('acc_*.tiff')))
  time.sleep(10)
  if jax.host_id() == 0 and num_files == dataset.size:
    print(f'All files found, creating videos.')
    create_videos(config, base_dir, out_dir, out_name, dataset.size)

  # A hack that forces Jax to keep all TPUs alive until every TPU is finished.
  x = jax.numpy.ones([jax.local_device_count()])
  x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
  print(x)
if __name__ == '__main__':
  app.run(main)
