from lpips import LPIPS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import compare_ssim, compare_psnr

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

lpips_vgg = None


@torch.no_grad()
def get_perceptual_metrics(rgbs, gts, lpips_batch_size=8, device='cuda'):
    # rgbs and gts should be numpy arrays of the same shape
    mse = img2mse(torch.from_numpy(rgbs), torch.from_numpy(gts)).item()

    # From pixelNeRF https://github.com/sxyu/pixel-nerf/blob/2929708e90b246dbd0329ce2a128ef381bd8c25d/eval/calc_metrics.py#L188
    global lpips_vgg
    ssim = [compare_ssim(rgb, gt, multichannel=True, data_range=1) for rgb, gt in zip(rgbs, gts)]
    ssim = np.mean(ssim)
    psnr = [compare_psnr(rgb, gt, data_range=1) for rgb, gt in zip(rgbs, gts)]
    psnr = np.mean(psnr)

    # From pixelNeRF https://github.com/sxyu/pixel-nerf/blob/2929708e90b246dbd0329ce2a128ef381bd8c25d/eval/calc_metrics.py#L238
    if lpips_vgg is None:
        lpips_vgg = LPIPS(net="vgg").to(device=device)
    lpips_all = []
    preds_spl = torch.split(torch.from_numpy(rgbs).permute(0,3,1,2).float(), lpips_batch_size, dim=0)
    gts_spl = torch.split(torch.from_numpy(gts).permute(0,3,1,2).float(), lpips_batch_size, dim=0)
    for predi, gti in zip(preds_spl, gts_spl):
        lpips_i = lpips_vgg(predi.to(device=device), gti.to(device=device))
        lpips_all.append(lpips_i)
    lpips = torch.cat(lpips_all)
    lpips = lpips.mean().item()

    return mse, psnr, ssim, lpips


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + W, W//2, activation="relu")])

        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            # TODO: use a softplus activation
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")

    def trunk_pts(self, input_pts):
        h = input_pts
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        return h

    def trunk_viewdirs(self, h):
        for layer in self.views_linears:
            h = layer(h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        return rgb

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = self.trunk_pts(input_pts)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            rgb = self.trunk_viewdirs(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


_printed_get_rays = False

# Ray helpers
def get_rays(H, W, focal, c2w, nH=None, nW=None, jitter=False):
    # nH and nW specify the number of rays for rows and columns of the rendered image, respectively
    # By setting nH < H or nW < W, we can render a smaller image that stretches the full
    # content extent of the scene
    if nH is None:
        nH = H
    if nW is None:
        nW = W

    if jitter:
        # Perturb query points
        dW = W // nW 
        # start_W = np.random.randint(low=0, high=W % nW + 1)
        start_W = np.random.uniform(low=0, high=W % nW)
        end_W = start_W + (nW - 1) * dW
        pts_W = torch.arange(start=start_W, end=end_W+1, step=dW)

        dH = H // nH
        # start_H = np.random.randint(low=0, high=H % nH + 1)
        start_H = np.random.uniform(low=0, high=H % nH)
        end_H = start_H + (nH - 1) * dH
        pts_H = torch.arange(start=start_H, end=end_H+1, step=dH)

        global _printed_get_rays
        if not _printed_get_rays:
            print('get_rays H', H)
            print('get_rays W', W)
            print('get_rays nH nW', nH, nW)
            print('get_rays pts_W', pts_W)
            print('get_rays pts_H', pts_H)
            _printed_get_rays = True
    else:
        pts_W = torch.linspace(0, W-1, nW)
        pts_H = torch.linspace(0, H-1, nH)
    i, j = torch.meshgrid(pts_W, pts_H)  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def gradient_norm(parameters):
    # https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
    total_norm = 0.
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
