import os
import random
import numpy as np
from PIL import Image

PATH = '/home/bratosiewicz/data/ffhq'
OUTPUT_FILE = 'out/ffhq_ref_batch.npz'
IMAGE_SIZE = (64, 64)

# Collect all image file paths
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
all_images = [
    os.path.join(root, file)
    for root, _, files in os.walk(PATH)
    for file in files
    if file.lower().endswith(image_extensions)
]

# Randomly sample 10000 images
sampled_images = random.sample(all_images, 10000)

# Load and preprocess images
images_array = []
for img_path in sampled_images:
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        img = img.resize(IMAGE_SIZE)
        images_array.append(np.array(img))

images_np = np.stack(images_array, axis=0)
np.savez(OUTPUT_FILE, arr_0=images_np)