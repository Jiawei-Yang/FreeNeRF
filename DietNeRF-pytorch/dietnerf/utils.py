import numpy as np
import torch


def get_freq_mask(embedded_shape, current_iter, total_iter):
    if len(embedded_shape) == 2:
        embedded_shape = embedded_shape[1]
    if current_iter > total_iter:
        return torch.ones(embedded_shape).cuda().float()
    else:
        alpha_t = np.zeros(embedded_shape)
        ptr = embedded_shape / 3 * current_iter / total_iter + 1 
        ptr = ptr if ptr < embedded_shape / 3 else embedded_shape / 3
        int_ptr = int(ptr)
        alpha_t[: int_ptr * 3] = 1.0
        alpha_t[int_ptr * 3: int_ptr * 3 + 3] = (ptr - int_ptr)
        return torch.from_numpy(np.clip(np.array(alpha_t), 1e-8, 1 - 1e-8)).cuda().float()
