import fire
import os

import numpy as np
import torch
import torchvision
import tqdm


from load_blender import load_blender_data


def probe(
    dataset_type,
    datadir,
    half_res,
    output_path,
    testskip=8,
    device='cuda',
    batch_size=16,
    model_type='clip_rn50',
):
    if dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(datadir, half_res, testskip)
        print('Loaded blender', images.shape, poses.shape, render_poses.shape, hwf, datadir)
        print('poses[0]', poses[0])
        print('render_poses[0]', render_poses[0])
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        raise NotImplementedError

    # Load embedding model
    if model_type.startswith('clip_'):
        import clip_utils

        normalize = clip_utils.CLIP_NORMALIZE
        if model_type == 'clip_rn50':
            clip_utils.load_rn()
            embed = lambda ims: clip_utils.clip_model_rn(images_or_text=ims)
            assert not clip_utils.clip_model_rn.training
        elif model_type == 'clip_vit':
            clip_utils.load_vit()
            embed = lambda ims: clip_utils.clip_model_vit(images_or_text=ims)[:, 0]  # select CLS token embedding, [N, D]
            assert not clip_utils.clip_model_vit.training
    elif model_type == 'crw_rn18':
        import crw_utils

        normalize = lambda ims: ims  # crw_utils.embed_image handles normalization
        crw_utils.load_rn18()
        embed = lambda ims: crw_utils.embed_image(ims, spatial_reduction='flatten')
        assert not crw_utils.crw_rn18_model.training
    elif model_type.startswith('imagenet_'):
        # Pretrained models in torchvision trained with ImageNet supervision
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        if model_type == 'imagenet_resnext50_32x4d':
            model = torchvision.models.resnext50_32x4d(pretrained=True)
            model.eval()
            model.to(device)
            # TODO: possibly set model.layer4 to identity to extract layer3 features?
            model.fc = torch.nn.Identity()  # don't project from features to classes
            embed = model
            assert not model.training
        # elif model_type == 'imagenet_wide_resnet50_2':
        #     pass

    # Prepare images
    images = torch.from_numpy(images).permute(0, 3, 1, 2)
    print('Loaded images:', images.shape, images.min(), images.max())

    # Embed images
    with torch.no_grad():
        # DEBUG: set some images to junk
        # images[-10:].uniform_()

        embedding = []
        for i in tqdm.trange(0, len(images), batch_size, desc='Embedding images'):
            images_batch = images[i:i+batch_size].float().to(device)
            images_batch = torch.nn.functional.interpolate(images_batch, size=(224, 224), mode='bicubic')
            images_batch = normalize(images_batch)
            print('images_batch', images_batch.shape)
            embedding_batch = embed(images_batch)
            embedding.append(embedding_batch)
        embedding = torch.cat(embedding, dim=0)
        print('Embedding:', embedding.shape)
        assert embedding.shape[0] == len(images)

        # Write results
        print('Saving embeddings to', output_path)
        torch.save({
            'images': images,
            'poses': poses,
            'render_poses': render_poses,
            'hwf': hwf,
            'i_split': i_split,
            'embedding': embedding.cpu().numpy(),
        }, output_path)


if __name__=='__main__':
    fire.Fire({
        'probe': probe
    })
