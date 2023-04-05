import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision

import clip


USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_NORMALIZE = torchvision.transforms.Normalize(CLIP_MEAN, CLIP_STD)  # normalize an image that is already scaled to [0, 1]


clip_model_vit, clip_model_rn = None, None
preprocess_vit, preprocess_rn = None, None


class CLIPEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images_or_text, encode_image=True, num_layers=-1, featurize=True):
        if encode_image:
            if featurize:
                images_or_text = images_or_text.type(self.clip_model.dtype)
                return self.clip_model.visual.featurize(images_or_text, num_layers=num_layers)

            assert num_layers == -1
            return self.clip_model.visual(images_or_text)

        return self.clip_model.encode_text(images_or_text)


def load_vit(name="ViT-B/32", jit=False, device=DEVICE, root: str=os.path.expanduser("~/.cache/clip")):
    global clip_model_vit, preprocess_vit
    if clip_model_vit is None:
        clip_model_vit, preprocess_vit = clip.load(name, device=device, jit=jit, root=root)  # jit=True doesn't seem to work with autodiff
        clip_model_vit.eval()
        clip_model_vit = torch.nn.DataParallel(CLIPEncoder(clip_model_vit))
        # clip_model_vit = CLIPEncoder(clip_model_vit)
        clip_model_vit.eval()
    return clip_model_vit, preprocess_vit


def load_rn(name='RN50', jit=False, device=DEVICE, root: str=os.path.expanduser("~/.cache/clip")):
    global clip_model_rn, preprocess_rn
    if clip_model_rn is None:
        clip_model_rn, preprocess_rn = clip.load(name, device=device, jit=jit, root=root)
        clip_model_rn.eval()
        clip_model_rn = torch.nn.DataParallel(CLIPEncoder(clip_model_rn))
        # clip_model_rn = CLIPEncoder(clip_model_rn)
        clip_model_rn.eval()
    return clip_model_rn, preprocess_rn


@torch.no_grad()
def embed_text(text: str):
    # Embed text
    assert isinstance(text, str)
    text = clip.tokenize(text).to(DEVICE)
    text_features_vit = clip_model_vit(text, encode_image=False)  # [1, 512]
    text_features_rn = clip_model_rn(text, encode_image=False)  # [1, 512]
    return torch.cat([text_features_vit, text_features_rn], dim=-1)


def rgba_to_rgb(rgba_image):
    # TODO: Try just taking the first 3 channels
    return rgba_image[:, :, 3:4] * rgba_image[:, :, :3] + torch.ones(rgba_image.shape[0], rgba_image.shape[1], 3, device=DEVICE) * (1 - rgba_image[:, :, 3:4])

    # # Composite rgba image with a white background (all 1s)
    # assert rgba_image.ndim == 3  # HWC
    # assert rgba_image.shape[2] == 4
    # rgb = rgba_image[..., :3]
    # alpha = rgba_image[..., 3:4]
    # return rgb * alpha + (1 - alpha)


def embed_image(image):
    # Convert and normalize image
    image = rgba_to_rgb(image)
    assert image.shape[0] == 224 and image.shape[1] == 224
    image = image.permute(2, 0, 1).unsqueeze(0)  # [224, 224, 3] to [1, 3, 224, 224]
    image = CLIP_NORMALIZE(image.to(DEVICE))

    # Embed
    features = []
    if clip_model_vit is not None:
        features.append(clip_model_vit(image))  # [1, 512]
    if clip_model_rn is not None:
        features.append(clip_model_rn(image))  # [1, 512]
    return torch.cat(features, dim=-1)


def plot_losses(losses, dir):
    plt.figure()
    plt.plot(-np.array(losses))
    plt.xlabel('Iteration')
    plt.ylabel('Cosine similarity')
    plt.savefig(os.path.join(dir, 'cosine_sim.pdf'))
    plt.savefig(os.path.join(dir, 'cosine_sim.png'))
