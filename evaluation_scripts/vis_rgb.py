import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import glob


def crop_image(img, crop_t, crop_b, crop_l, crop_r):
    """
    Crop image from (crop_t, crop_l) to (crop_b, crop_r)
    crop_l, crop_r: left and right boundaries
    crop_t, crop_b: top and bottom boundaries
    """
    return img[crop_t:crop_b, crop_l:crop_r]



def main():
    original_rgb = "/workspace/dataset/StereoMIS/P2_8/left_finalpass/*.png"
    save_rgb_path = "/workspace/StereoMamba2/cropped_rgbimg/stereomis/P2_8"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_rgb_path):
        os.makedirs(save_rgb_path, exist_ok=True)

    
    # Process all disparity maps
    for rgb_file in glob.glob(original_rgb):
        # Load disparity map (remove the division by 128.0)
        img = np.array(Image.open(rgb_file))
        # import ipdb; ipdb.set_trace()
        # img = crop_image(img, 45, 980, 93, 1273) # Crop the scared
        img = crop_image(img, 5, 1020, 48, 1280) # Crop the stereomis

        
        # Save colored disparity map
        save_name = os.path.join(save_rgb_path, os.path.basename(rgb_file))
        Image.fromarray(img).save(save_name)
        
    print(f"Processed disparity maps saved to {save_rgb_path}")

if __name__ == "__main__":
    main()

