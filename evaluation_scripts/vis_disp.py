import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import glob


DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def crop_image(img, crop_t, crop_b, crop_l, crop_r):
    """
    Crop image from (crop_t, crop_l) to (crop_b, crop_r)
    crop_l, crop_r: left and right boundaries
    crop_t, crop_b: top and bottom boundaries
    """
    return img[crop_t:crop_b, crop_l:crop_r]

def disp_to_color(disp, min_disp=0, max_disp=192):
    disp = np.squeeze(disp)
    # disp = np.clip(disp, min_disp, max_disp)
    disp = disp / 128.0
    # Normalize values for colormap (0-1 range)
    disp_norm = (disp - min_disp) / (max_disp - min_disp)
    # import ipdb; ipdb.set_trace()
    # Create heatmap
    cmap = plt.get_cmap('jet')
    colored_disp = cmap(disp_norm)[:, :, :3]  # Remove alpha channel
    return (colored_disp * 255).astype(np.uint8)

def generate_colorbar(save_path, min_disp=0, max_disp=192):
    """Generate a vertical colorbar and save it"""
    fig, ax = plt.subplots(figsize=(2, 8))
    norm = plt.Normalize(min_disp, max_disp)
    cmap = plt.get_cmap('jet')
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                     cax=ax, orientation='vertical')
    cb.set_label('Disparity', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    disp_map_path = "/workspace/StereoMamba2/cropped_rgbimg/stereomis/P2_2/*.png"
    save_color_path = "/workspace/dataset/RAL_video/rgb/stereomis_22.mp4"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_color_path):
        os.makedirs(save_color_path, exist_ok=True)
    
    # Generate colorbar
    # colorbar_path = os.path.join(save_color_path, 'colorbar.png')
    # generate_colorbar(colorbar_path)
    
    # Process all disparity maps
    for disp_file in glob.glob(disp_map_path):
        # Load disparity map (remove the division by 128.0)
        disp = np.array(Image.open(disp_file))
        # import ipdb; ipdb.set_trace()
        # disp = crop_image(disp, 45, 980, 93, 1273) # crop the scared
        disp = crop_image(disp, 5, 1020, 48, 1280) # crop the stereomis
        # Convert to heatmap
        colored_disp = disp_to_color(disp)
        
        # Save colored disparity map
        save_name = os.path.join(save_color_path, os.path.basename(disp_file))
        Image.fromarray(colored_disp).save(save_name)
        
    print(f"Processed disparity maps saved to {save_color_path}")

if __name__ == "__main__":
    main()

