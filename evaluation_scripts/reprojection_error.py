import os
import numpy as np
import torch
import lpips
import cv2
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = torch.mean(x)
    mu_y = torch.mean(y)
    
    sigma_x = torch.std(x)
    sigma_y = torch.std(y)
    
    sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
    
    L = 1
    C3 = (C2 / 2) ** 2
    
    l_xy = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c_xy = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
    s_xy = (sigma_xy + C3) / (sigma_x * sigma_y + C3)
    
    return l_xy * c_xy * s_xy

def PSNR(x, y):
    mse = torch.mean((x - y) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))  

def LPIPS(x, y, loss_fn):
    
    return loss_fn.forward(x, y)


def compute_errors(gt, pred, loss_fn):
    # Convert HWC to BCHW format
    gt = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).float().to('cuda')
    pred = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0).float().to('cuda')

    # LPIPS expects values in [-1, 1]
    gt = gt * 2 - 1
    pred = pred * 2 - 1

    error_ssim = SSIM(gt.squeeze(), pred.squeeze())
    error_psnr = PSNR(gt.squeeze(), pred.squeeze())
    error_lpips = LPIPS(gt, pred, loss_fn)
    return error_ssim, error_psnr, error_lpips

def transform_image(left_img, disparity):
    h, w = left_img.shape[:2]
    
    # Create meshgrid for coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    # Calculate new x coordinates based on disparity
    new_x = x_coords + disparity
    
    # Create lookup table for warping
    map_x = new_x.astype(np.float32)
    map_y = y_coords.astype(np.float32)
    
    # Warp the image
    right_img = cv2.remap(left_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    return right_img


left_img_path = '/workspace/dataset/ris_2017/train_set/instrument_dataset_8/left_frame'
right_img_path = '/workspace/dataset/ris_2017/train_set/instrument_dataset_8/right_frame'
disp_path = '/workspace/StereoMamba2/disparity_maps_scared_epoch_100/ris2017_trainset/train_set/instrument_dataset_8'

def main():
    # Create output directory for warped images
    output_dir = os.path.join(os.path.dirname(disp_path), 'warped_images')
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    left_files = sorted(glob.glob(os.path.join(left_img_path, '*.png')))
    right_files = sorted(glob.glob(os.path.join(right_img_path, '*.png')))
    disp_files = sorted(glob.glob(os.path.join(disp_path, '*.png')))
    
    avg_ssim = 0
    avg_psnr = 0
    avg_lpips = 0
    count = 0
    loss_fn = lpips.LPIPS(net='vgg').to('cuda')
    for left_file, right_file, disp_file in zip(left_files, right_files, disp_files):
        # Read images
        left_img = cv2.imread(left_file)
        right_img = cv2.imread(right_file)
        disparity = cv2.imread(disp_file, cv2.IMREAD_GRAYSCALE) * 2.0 
        
        # Transform left image to right view
        synth_right = transform_image(left_img, disparity)
        
        # Save warped image
        # import ipdb; ipdb.set_trace()
        output_filename = os.path.join(output_dir, os.path.basename(right_file))
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        cv2.imwrite(output_filename, synth_right)

        # Convert to format required by error metrics
        right_img = right_img / 255.0
        synth_right = synth_right / 255.0
        
        # Compute errors
        ssim, psnr, lpips_val = compute_errors(right_img, synth_right, loss_fn)
        
        avg_ssim += ssim.item()
        avg_psnr += psnr.item()
        avg_lpips += lpips_val.item()
        count += 1
        
        print(f"Image {count}: SSIM: {ssim.item():.4f}, PSNR: {psnr.item():.4f}, LPIPS: {lpips_val.item():.4f}")
    
    # Print average metrics
    print(f"\nAverage metrics over {count} images:")
    print(f"SSIM: {avg_ssim/count:.4f}")
    print(f"PSNR: {avg_psnr/count:.4f}")
    print(f"LPIPS: {avg_lpips/count:.4f}")

if __name__ == "__main__":
    main()