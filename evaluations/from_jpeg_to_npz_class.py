import numpy as np
import glob
import os
import blobfile as bf
from PIL import Image
import torch
from torchvision.utils import make_grid, save_image

sampler = 'ddpm'
dir = '221003_original_npz'
save_dir_ = '/data/NinthArticleExperimentalResults/ImageNet32/training_data/'
img_dir = '/data/data/ImageNet'
ext = 'JPEG'
resolution = 32
class_idx = 0

def center_crop_arr(pil_image, image_size, data_name='cifar10'):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    if data_name in ['church']:
        img = np.array(pil_image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if image_size is not None:
            image = image.resize((image_size, image_size), resample='bicubic')
        return image
    else:
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


dirs = glob.glob(os.path.join(img_dir, '*'))
#print("dirs: ", dirs)
arrs = []

dir_ = np.sort(dirs)[class_idx]
#print("dir_: ", dir_)
filelist = glob.glob(os.path.join(dir_, f'*.{ext}'))
for path in filelist:
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = center_crop_arr(pil_image, resolution).reshape(1,resolution,resolution,3)
    arrs.append(arr)
arrs = np.concatenate(arrs, 0)
arr = torch.tensor(arrs).permute(0,3,1,2)/255.
nrow = int(np.sqrt(arr.shape[0]))
image_grid = make_grid(arr, nrow, padding=2)
with bf.BlobFile(bf.join(save_dir_, f"class_{class_idx}.png"), "wb") as fout:
    save_image(image_grid, fout)
print(arrs.shape)