import json
import os
import time

import clip_utils
import configargparse
import imageio
import numpy as np
from scipy.spatial.transform import Rotation
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as run_checkpoint
import torchvision
from tqdm import tqdm, trange
import wandb

import geometry
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, pose_spherical_uniform
from run_nerf_helpers import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, keep_keys=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    # ret_rgb_only = keep_keys and len(keep_keys) == 1 and keep_keys[0] == 'rgb_map'
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if keep_keys and k not in keep_keys:
                # Don't save this returned value to save memory
                continue
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  keep_keys=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, keep_keys=keep_keys, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    if keep_keys:
        k_extract = [k for k in k_extract if k in keep_keys]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    model = nn.DataParallel(model).to(device)
    wandb.watch(model)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        model_fine = nn.DataParallel(model_fine).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=args.render_autocast)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None' and not args.render_only:
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        if args.reload_iter:
            print('Trying to reload a specific iteration:', args.reload_iter)
            ckpts = list(filter(lambda ckpt: ckpt.endswith(f'/{args.reload_iter}.tar'), ckpts))
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################


    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'alpha_act_fn' : F.softplus if args.use_softplus_alpha else F.relu
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, scaler


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, alpha_act_fn=F.relu):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists: 1.-torch.exp(-alpha_act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                verbose=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                pytest=False,
                alpha_act_fn=F.relu):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, alpha_act_fn=alpha_act_fn)

    if N_importance > 0:
        pts0, raw0, rgb_map_0, disp_map_0, acc_map_0, weights0 = pts, raw, rgb_map, disp_map, acc_map, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, alpha_act_fn=alpha_act_fn)

    ret = {'rgb_map' : rgb_map}
    ret['disp_map'] = disp_map
    ret['acc_map'] = acc_map
    ret['weights'] = weights
    ret['pts'] = pts
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['raw0'] = raw0
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['pts0'] = pts0
        ret['weights0'] = weights0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def sample_rays(H, W, rays_o, rays_d, N_rand, i, start, precrop_iters, precrop_frac):
    if i < precrop_iters:
        dH = int(H//2 * precrop_frac)
        dW = int(W//2 * precrop_frac)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            ), -1)
        if i == start:
            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {precrop_iters}")                
    else:
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    return batch_rays, select_coords


def get_embed_fn(model_type, num_layers=-1, spatial=False, checkpoint=False, clip_cache_root=None):
    if model_type.startswith('clip_'):
        if model_type == 'clip_rn50':
            assert clip_cache_root
            clip_utils.load_rn(jit=False, root=clip_cache_root)
            if spatial:
                _clip_dtype = clip_utils.clip_model_rn.clip_model.dtype
                assert num_layers == -1
                def embed(ims):
                    ims = clip_utils.CLIP_NORMALIZE(ims).type(_clip_dtype)
                    return clip_utils.clip_model_rn.clip_model.visual.featurize(ims)  # [N,C,56,56]
            else:
                embed = lambda ims: clip_utils.clip_model_rn(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers).unsqueeze(1)
            assert not clip_utils.clip_model_rn.training
        elif model_type.startswith('clip_vit'):
            assert clip_cache_root
            if model_type == 'clip_vit':
                clip_utils.load_vit(root=clip_cache_root)
            elif model_type == 'clip_vit_b16':
                clip_utils.load_vit('ViT-B/16', root=clip_cache_root)
            if spatial:
                def embed(ims):
                    emb = clip_utils.clip_model_vit(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers)  # [N,L=50,D]
                    return emb[:, 1:].view(emb.shape[0], 7, 7, emb.shape[2]).permute(0, 3, 1, 2)  # [N,D,7,7]
            else:
                embed = lambda ims: clip_utils.clip_model_vit(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers)  # [N,L=50,D]
            assert not clip_utils.clip_model_vit.training
        elif model_type == 'clip_rn50x4':
            assert not spatial
            clip_utils.load_rn(name='RN50x4', jit=False)
            assert not clip_utils.clip_model_rn.training
            embed = lambda ims: clip_utils.clip_model_rn(images_or_text=clip_utils.CLIP_NORMALIZE(ims), featurize=False)
    elif model_type.startswith('timm_'):
        assert num_layers == -1
        assert not spatial

        model_type = model_type[len('timm_'):]
        encoder = timm.create_model(model_type, pretrained=True, num_classes=0)
        encoder.eval()
        normalize = torchvision.transforms.Normalize(
            encoder.default_cfg['mean'], encoder.default_cfg['std'])  # normalize an image that is already scaled to [0, 1]
        encoder = nn.DataParallel(encoder).to(device)
        embed = lambda ims: encoder(normalize(ims)).unsqueeze(1)
    elif model_type.startswith('torch_'):
        assert num_layers == -1
        assert not spatial

        model_type = model_type[len('torch_'):]
        encoder = torch.hub.load('pytorch/vision:v0.6.0', model_type, pretrained=True)
        encoder.eval()
        encoder = nn.DataParallel(encoder).to(device)
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize an image that is already scaled to [0, 1]
        embed = lambda ims: encoder(normalize(ims)).unsqueeze(1)
    else:
        raise ValueError

    if checkpoint:
        return lambda x: run_checkpoint(embed, x)

    return embed


@torch.no_grad()
def make_wandb_image(tensor, preprocess='scale'):
    tensor = tensor.detach()
    tensor = tensor.float()
    if preprocess == 'scale':
        mi = tensor.min()
        tensor = ((tensor - mi) / (tensor.max() - mi))
    elif preprocess == 'clip':
        tensor = tensor.clip(0, 1)
    return wandb.Image(tensor.cpu().numpy())


@torch.no_grad()
def make_wandb_histogram(tensor):
    return wandb.Histogram(tensor.detach().flatten().cpu().numpy())


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024*64*4, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--reload_iter", type=str, default=None, 
                        help='load a specific iteration rather than the latest. will load expdir/<reload_iter>.tar')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_softplus_alpha", action='store_true',
                        help='use a softplus activation on predicted alphas rather than relu')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--N_iters", type=int, default=200000)

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--full_res", action='store_false', dest='half_res',
                        help='load blender synthetic data at 800x800')
    parser.add_argument("--num_render_poses", type=int, default=40)

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--wandb_project",   type=str, default='dietnerf')
    parser.add_argument("--i_log",   type=int, default=1, 
                        help='frequency of metric logging')
    parser.add_argument("--i_log_raw_hist",   type=int, default=2, 
                        help='frequency of logging histogram of raw network outputs')
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--save_splits", action="store_true",
                        help='save ground truth images and poses in each split')

    ### options for learning with few views
    parser.add_argument("--max_train_views", type=int, default=-1,
                        help='limit number of training views for the mse loss')
    parser.add_argument("--hardcode_train_views", type=int, nargs="+", default=[])
    # Options for rendering shared between different losses
    parser.add_argument("--render_loss_interval", "--consistency_loss_interval",
        type=float, default=1)
    parser.add_argument("--render_autocast", action='store_true')
    parser.add_argument("--render_poses", "--consistency_poses",
        type=str, choices=['loaded', 'interpolate_train_all', 'uniform'], default='loaded')
    parser.add_argument("--render_poses_translation_jitter_sigma", "--consistency_poses_translation_jitter_sigma",
        type=float, default=0.)
    parser.add_argument("--render_poses_interpolate_range", "--consistency_poses_interpolate_range",
        type=float, nargs=2, default=[0., 1.])
    # Options for --render_poses=uniform
    parser.add_argument("--render_theta_range", "--consistency_theta_range", type=float, nargs=2)
    parser.add_argument("--render_phi_range", "--consistency_phi_range", type=float, nargs=2)
    parser.add_argument("--render_radius_range", "--consistency_radius_range", type=float, nargs=2)
    parser.add_argument("--render_nH", "--consistency_nH", type=int, default=32, 
                        help='number of rows to render for consistency loss. smaller values use less memory')
    parser.add_argument("--render_nW", "--consistency_nW", type=int, default=32, 
                        help='number of columns to render for consistency loss')
    parser.add_argument("--render_jitter_rays", "--consistency_jitter_rays", action='store_true')

    # Computational options for rendering losses
    parser.add_argument("--checkpoint_rendering", action='store_true')
    parser.add_argument("--checkpoint_embedding", action='store_true')
    parser.add_argument("--no_mse", action='store_true')
    parser.add_argument("--pixel_interp_mode", type=str, default='bicubic')
    parser.add_argument("--feature_interp_mode", type=str, default='bilinear')
    # Semantic consistency loss
    parser.add_argument("--consistency_loss", type=str, default='none', choices=['none', 'consistent_with_target_rep'])
    parser.add_argument("--consistency_loss_lam", type=float, default=0.2,
                        help="weight for the fine network's semantic consistency loss")
    parser.add_argument("--consistency_loss_lam0", type=float, default=0.2,
                        help="weight for the coarse network's semantic consistency loss")
    parser.add_argument("--consistency_size", type=int, default=224)
    # Consistency model arguments
    parser.add_argument("--consistency_model_type", type=str, default='clip_vit') # choices=['clip_vit', 'clip_vit_b16', 'clip_rn50']
    parser.add_argument("--consistency_model_num_layers", type=int, default=-1)
    parser.add_argument("--clip_cache_root", type=str, default=os.path.expanduser("~/.cache/clip"))

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.run.name = args.expname
    wandb.run.save()
    wandb.config.update(args)

    # Re-seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    print('dataset_type:', args.dataset_type)
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip, num_render_poses=args.num_render_poses)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Subsample training indices to simulate having fewer training views
    i_train_poses = i_train  # Use all training poses for auxiliary representation consistency loss.
                             # TODO: Could also use any continuous set of poses including the
                             # val and test poses, since we don't use the images for aux loss.
    if len(args.hardcode_train_views):
        print('Original training views:', i_train)
        i_train = np.array(args.hardcode_train_views)
        print('Hardcoded train views:', i_train)
    elif args.max_train_views > 0:
        print('Original training views:', i_train)
        i_train = np.random.choice(i_train, size=args.max_train_views, replace=False)
        print('Subsampled train views:', i_train)

    # Load embedding network for rendering losses
    if args.consistency_loss != 'none':
        print(f'Using auxilliary consistency loss [{args.consistency_loss}], fine weight [{args.consistency_loss_lam}], coarse weight [{args.consistency_loss_lam0}]')
        embed = get_embed_fn(args.consistency_model_type, args.consistency_model_num_layers, checkpoint=args.checkpoint_embedding, clip_cache_root=args.clip_cache_root)

    # Cast intrinsics to right types
    H, W, focal = hwf
    print('hwf', hwf)
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, scaler = create_nerf(args)
    global_step = start
    network_fn = render_kwargs_train['network_fn']
    network_fine = render_kwargs_train['network_fine']
    if args.checkpoint_rendering:
        # Pass a dummy input tensor that requires grad so checkpointing does something
        # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/10
        dummy = torch.ones(1, dtype=torch.float32, requires_grad=True, device=device)
        network_fn_wrapper = lambda x, y: network_fn(x)
        network_fine_wrapper = lambda x, y: network_fine(x)
        render_kwargs_train['network_fn'] = lambda x: run_checkpoint(network_fn_wrapper, x, dummy)
        render_kwargs_train['network_fine'] = lambda x: run_checkpoint(network_fine_wrapper, x, dummy)


    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            if args.render_test:
                # Compute metrics
                mse, psnr, ssim, lpips = get_perceptual_metrics(rgbs, images, device=device)

                metricspath = os.path.join(testsavedir, 'test_metrics.json')
                with open(metricspath, 'w') as test_metrics_f:
                    test_metrics = {
                        'mse': mse,
                        'psnr': psnr,
                        'ssim': ssim,
                        'lpips': lpips,
                    }
                    print(args.expname, f'test metrics ({metricspath}):', test_metrics)
                    test_metrics['args'] = vars(args)
                    json.dump(test_metrics, test_metrics_f)
                    wandb.save(metricspath)

            return

    # Save ground truth splits for visualization
    if args.save_splits:
        for idx, name in [(i_train, 'train'), (i_val, 'val'), (i_test, 'test')]:
            savedir = os.path.join(basedir, expname, '{}set'.format(name))
            os.makedirs(savedir, exist_ok=True)

            torch.save(poses[idx], os.path.join(savedir, 'poses.pth'))
            torch.save(idx, os.path.join(savedir, 'indices.pth'))
            for i in idx:
                rgb8 = to8b(images[i])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

            print(name, 'poses shape', poses[idx].shape, 'images shape', images[idx].shape)
            print(f'Saved ground truth {name} set')

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        assert not args.render_jitter_rays
        print('get rays')

        # For random ray batching, equal number of rays per pose
        rays_rgb_o = []
        rays_rgb_d = []
        for p in poses[i_train,:3,:4]:
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(p))  # (H, W, 3), (H, W, 3)
            rays_rgb_o.append(rays_o)
            rays_rgb_d.append(rays_d)
        N_rand_per_view = N_rand // len(i_train)

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    calc_ctr_loss = args.consistency_loss.startswith('consistent_with_target_rep')
    any_rendered_loss = calc_ctr_loss
    if any_rendered_loss:
        with torch.no_grad():
            targets = images[i_train].permute(0, 3, 1, 2).to(device)

    # Embed training images for consistency loss
    if calc_ctr_loss:
        with torch.no_grad():
            targets_resize_model = F.interpolate(targets, (args.consistency_size, args.consistency_size), mode=args.pixel_interp_mode)
            target_embeddings = embed(targets_resize_model)  # [N,L,D]

    # Embed training images for aligned consistency loss
    consistency_keep_keys = ['rgb_map', 'rgb0']
 
    start = start + 1
    for i in trange(start, N_iters):
        metrics = {}

        # Sample random ray batch
        if use_batching:
            # Select the same number of rays for each view
            batch_rays, target_s = [], []
            for target, rays_o, rays_d in zip(images[i_train], rays_rgb_o, rays_rgb_d):
                rays_pose, select_coords = sample_rays(H, W, rays_o, rays_d, N_rand=N_rand_per_view,
                    i=i, start=start, precrop_iters=args.precrop_iters, precrop_frac=args.precrop_frac)
                batch_rays.append(rays_pose)
                target_s_pose = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                target_s.append(target_s_pose)
            batch_rays = torch.cat(batch_rays, dim=1).to(device)  # (2, N_rand, 3)
            target_s = torch.cat(target_s, dim=0).to(device)  # (N_rand, 3)
        else:
            assert N_rand is not None

            # Random rays from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            batch_rays, select_coords = sample_rays(H, W, rays_o, rays_d, N_rand=N_rand,
                i=i, start=start, precrop_iters=args.precrop_iters, precrop_frac=args.precrop_frac)
            batch_rays = batch_rays.to(device)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            target_s = target_s.to(device)

        # Representational consistency loss with rendered image
        render_loss_iter = i % args.render_loss_interval == 0

        if any_rendered_loss and render_loss_iter:
            with torch.no_grad():
                # Render from a random viewpoint
                if args.render_poses == 'loaded':
                    poses_i = np.random.choice(i_train_poses)
                    pose = poses[poses_i, :3, :4]
                elif args.render_poses == 'interpolate_train_all':
                    assert len(i_train_poses) >= 3
                    poses_i = np.random.choice(i_train_poses, size=3, replace=False)
                    pose1, pose2, pose3 = poses[poses_i, :3, :4].cpu()
                    s12, s3 = np.random.uniform(*args.render_poses_interpolate_range, size=2)
                    pose = geometry.interp3(pose1, pose2, pose3, s12, s3)
                elif args.render_poses == 'uniform':
                    assert args.dataset_type == 'blender'
                    pose = pose_spherical_uniform(args.render_theta_range, args.render_phi_range, args.render_radius_range)
                    pose = pose[:3, :4]

                print('Sampled pose:', Rotation.from_matrix(pose[:, :3].cpu()).as_rotvec(), 'origin:', pose[:, 3])

                if args.render_poses_translation_jitter_sigma > 0:
                    pose[:, -1] = pose[:, -1] + torch.randn(3, device=pose.device) * args.render_poses_translation_jitter_sigma

                # TODO: something strange with pts_W in get_rays when 224 nH
                rays = get_rays(H, W, focal, c2w=pose, nH=args.render_nH, nW=args.render_nW,
                                jitter=args.render_jitter_rays)

            with torch.cuda.amp.autocast(enabled=args.render_autocast):
                extras = render(H, W, focal, chunk=args.chunk,
                                rays=(rays[0].to(device), rays[1].to(device)),
                                keep_keys=consistency_keep_keys,
                                **render_kwargs_train)[-1]
                # rgb0 is the rendering from the coarse network, while rgb_map uses the fine network
                if args.N_importance > 0:
                    rgbs = torch.stack([extras['rgb_map'], extras['rgb0']], dim=0)
                else:
                    rgbs = extras['rgb_map'].unsqueeze(0)
                rgbs = rgbs.permute(0, 3, 1, 2).clamp(0, 1)

            if i == 0:
                print('rendering losses rendered rgb image shape:', rgbs.shape)

            # Log rendered images
            metrics['train_ctr/rgb'] = make_wandb_image(extras['rgb_map'], 'clip')
            if args.N_importance > 0:
                metrics['train_ctr/rgb0'] = make_wandb_image(extras['rgb0'], 'clip')

        #####  Core optimization loop  #####

        optimizer.zero_grad()
        loss = 0

        if calc_ctr_loss and render_loss_iter:
            assert args.consistency_loss == 'consistent_with_target_rep'

            # Resize and embed rendered images
            rgbs_resize_c = F.interpolate(rgbs, size=(args.consistency_size, args.consistency_size), mode=args.pixel_interp_mode)
            rendered_embeddings = embed(rgbs_resize_c)
            rendered_embedding = rendered_embeddings[0]
            if args.N_importance > 0:
                rendered_embedding0 = rendered_embeddings[1]  # for coarse net

            # Randomly sample a target
            if args.consistency_model_type.startswith('clip_vit'):
                # Modified CLIP ViT to return sequence features. Extract the [CLS] token features
                assert rendered_embedding.ndim == 2  # [L,D]
                assert target_embeddings.ndim == 3  # [N,L,D]
                rendered_emb = rendered_embedding[0]
                if args.N_importance > 0:
                    rendered_emb0 = rendered_embedding0[0]
                target_emb = target_embeddings[:, 0]
            else:
                assert rendered_embedding.ndim == 1  # [D]
                assert target_embeddings.ndim == 2  # [N,D]
                rendered_emb, target_emb = rendered_embedding, target_embeddings
                if args.N_importance > 0:
                    rendered_emb0 = rendered_embedding0

            # Sample a single random target for consistency loss
            target_i = np.random.randint(target_emb.shape[0])
            target_emb = target_emb[target_i]
            consistency_loss = -torch.cosine_similarity(target_emb, rendered_emb, dim=-1)
            if args.N_importance > 0:
                consistency_loss0 = -torch.cosine_similarity(target_emb, rendered_emb0, dim=-1)

            loss = loss + consistency_loss * args.consistency_loss_lam
            if args.N_importance > 0:
                loss = loss + consistency_loss0 * args.consistency_loss_lam0

        if not args.no_mse:
            # Standard NeRF MSE loss with subsampled rays
            rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True,
                                            **render_kwargs_train)
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = loss + img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                if i == start:
                    print('Using auxilliary rgb0 mse loss')
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': network_fn.state_dict(),
                'network_fine_state_dict': network_fine.state_dict() if network_fine is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            wandb.save(path)
            print('Uploading checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
                metrics["render_path/rgb_video"] = wandb.Video(moviebase + 'rgb.mp4')
                metrics["render_path/disp_video"] = wandb.Video(moviebase + 'disp.mp4')

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        # Log scalars, images and histograms to wandb
        if i%args.i_log==0:
            metrics.update({
                "train/loss": loss.item(),
                "train/psnr": psnr.item(),
                "train/mse": img_loss.item(),
                "train/lrate": new_lrate,
            })
            metrics["gradients/norm_coarse"] = gradient_norm(network_fn.parameters())
            if args.N_importance > 0:
                metrics["gradients/norm_fine"] = gradient_norm(network_fine.parameters())
                metrics["train/psnr0"] = psnr0.item()
                metrics["train/mse0"] = img_loss0.item()
            if render_loss_iter and calc_ctr_loss:
                metrics["train_ctr/consistency_loss"] = consistency_loss.item()
                if args.N_importance > 0:
                    metrics["train_ctr/consistency_loss0"] = consistency_loss0.item()

        if i%args.i_log_raw_hist==0:
            metrics["train/tran"] = wandb.Histogram(trans.detach().cpu().numpy())

        if i%args.i_img==0:
            # Log a rendered validation view to Tensorboard
            with torch.no_grad():
                img_i = i_val[0]
                target = images[img_i]
                pose = poses[img_i, :3,:4].to(device)
                rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                    **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                metrics = {
                    'val/rgb': wandb.Image(to8b(rgb.cpu().numpy())[np.newaxis]),
                    'val/disp': wandb.Image(disp.cpu().numpy()[np.newaxis,...,np.newaxis]),
                    'val/disp_scaled': make_wandb_image(disp[np.newaxis,...,np.newaxis]),
                    'val/acc': wandb.Image(acc.cpu().numpy()[np.newaxis,...,np.newaxis]),
                    'val/acc_scaled': make_wandb_image(acc[np.newaxis,...,np.newaxis]),
                    'val/psnr_holdout': psnr.item(),
                    'val/rgb_holdout': wandb.Image(target.cpu().numpy()[np.newaxis])
                }
                if args.N_importance > 0:
                    metrics['rgb0'] = wandb.Image(to8b(extras['rgb0'].cpu().numpy())[np.newaxis])
                    metrics['disp0'] = wandb.Image(extras['disp0'].cpu().numpy()[np.newaxis,...,np.newaxis])
                    metrics['z_std'] = wandb.Image(extras['z_std'].cpu().numpy()[np.newaxis,...,np.newaxis])

        if metrics:
            wandb.log(metrics, step=i)

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
