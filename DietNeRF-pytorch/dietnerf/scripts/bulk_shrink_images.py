import os
import fire
import glob
import sys
import tqdm
from PIL import Image
import numpy as np
import ray

def alpha_composite(front, back):
    """Alpha composite two RGBA images.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object

    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result

def alpha_composite_with_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)

def shrink(test_img):
    test_img = np.asarray(test_img).astype(np.float64)
    # print(test_img.dtype, test_img.min(), test_img.max())
    test_img = (test_img[::2,::2,:] + test_img[1::2,::2,:] + test_img[1::2,1::2,:] + test_img[::2,1::2,:]) / 4
    # print(test_img.dtype, test_img.min(), test_img.max())
    test_img = test_img.astype(np.uint8)
    test_img = Image.fromarray(test_img)
    if len(test_img.getbands()) != 3:
        test_img = alpha_composite_with_color(test_img)
    return test_img

def do_resize(input_glob_pattern, output_directory, overwrite=False):
    if os.path.exists(output_directory) and not overwrite:
        print("Output directory", output_directory, "already exists. Exiting.")
        sys.exit(1)

    os.makedirs(output_directory, exist_ok=overwrite)

    ray.init(num_cpus=16)

    @ray.remote
    def _resize(path):
        out_path = path.replace("/", "-")
        out_path = os.path.join(output_directory, out_path)
        img = Image.open(path)
        img_small = shrink(img)
        img_small.save(out_path)

    input_paths = glob.glob(input_glob_pattern)
    futures = []
    for path in tqdm.tqdm(input_paths):
        if "normal" not in path and "depth" not in path:
            f = _resize.remote(path)
            futures.append(f)
    ray.get(futures)

if __name__=="__main__":
    fire.Fire(do_resize)