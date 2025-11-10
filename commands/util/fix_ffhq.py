import os
import shutil
import re

# Paths
input_dir = "/home/bratosiewicz/pagoda/out/ffhq-xz-flip/edm_heun_sampler_40_steps_060000_itrs_0.9999_ema_7_rho"  # adjust if needed
output_dir = "/home/bratosiewicz/pagoda/out/ffhq-xz-flip/train"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Regex pattern to match files
pattern = re.compile(r"sample_(\d+)\.npz")

for filename in os.listdir(input_dir):
    match = pattern.match(filename)
    if match:
        num = int(match.group(1))
        new_num = num + 2000 if num >= 9000 else num
        new_filename = f"{new_num:05d}.npz"
        
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, new_filename)
        
        shutil.copy2(src_path, dst_path)  # copy with metadata

print("Renaming complete.")