import argparse
from pathlib import Path
import torch
import json
from utils.error_metrics import disparity_bad3, disparity_epe
import numpy as np
import pandas as pd
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def load_disparity(imfile):
    disp = np.array(Image.open(imfile)).astype(np.float32)
    return torch.from_numpy(disp)[None, None].cuda()

def load_calib(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    baseline = abs(calib['T']['data'][0])  # in mm
    focal_length = calib['P1']['data'][0]  # in pixels
    return baseline, focal_length

parser = argparse.ArgumentParser('Evaluate disparity/depth on SCARED dataset')
parser.add_argument('--predictions', default='/workspace/StereoMamba/output_scared')
parser.add_argument('--ground_truth', default='/workspace/dataset/scared/test_data')
parser.add_argument('--csv', default='/workspace/StereoMamba/output_scared/scared.csv')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize keyframe-level metrics storage
keyframe_metrics = {
    'dataset': [], 'keyframe': [],
    'EPE': [], 'Bad1': [], 'Bad2': [], 'Bad3': [], 'Bad5': [], 'Depth_MAE': []
}

# Process datasets 8 and 9
for dataset in ['dataset_8', 'dataset_9']:
    # Process keyframes 0 to 5
    for keyframe in range(6):
        keyframe_dir = f'keyframe_{keyframe}'
        base_gt_path = Path(args.ground_truth) / dataset / keyframe_dir
        base_pred_path = Path(args.predictions) / dataset / keyframe_dir
        calib_path = base_gt_path / 'stereo_calib.json'
        
        if not calib_path.exists():
            continue
            
        baseline, focal_length = load_calib(str(calib_path))
        
        # Initialize accumulators for this keyframe
        epe_sum = 0.0
        bad1_sum = 0.0
        bad2_sum = 0.0
        bad3_sum = 0.0
        bad5_sum = 0.0
        depth_mae_sum = 0.0
        frame_count = 0
        
        # Process all frames in the keyframe
        gt_disp_path = base_gt_path / 'data/disparity'
        for gt_file, pred_file in zip(sorted(gt_disp_path.glob('*.png')), sorted(base_pred_path.glob('*.png'))):

            if not pred_file.exists():
                continue

            # Load disparities
            disp_gt = load_disparity(gt_file) / 128.0
            disp_pred = load_disparity(pred_file) / 128.0
            
            disp_gt = disp_gt.squeeze(0)
            disp_pred = disp_pred.squeeze(0)

            if disp_gt.count_nonzero()/torch.numel(disp_gt)<0.1:
                print("discard frame", gt_file.name)
                continue
                
            valid_mask = disp_gt > 0
            
            # Calculate metrics
            epe = disparity_epe(disp_pred, disp_gt, max_disparity=192)
            bad1 = disparity_bad3(disp_pred, disp_gt, max_disparity=192, thres=1)
            bad2 = disparity_bad3(disp_pred, disp_gt, max_disparity=192, thres=2)
            bad3 = disparity_bad3(disp_pred, disp_gt, max_disparity=192, thres=3)
            bad5 = disparity_bad3(disp_pred, disp_gt, max_disparity=192, thres=5)
            
            depth_pred = baseline * focal_length / (disp_pred + 1e-6)
            depth_gt = baseline * focal_length / (disp_gt + 1e-6)
            depth_error = torch.abs(depth_pred - depth_gt) * valid_mask.float()
            depth_mae = depth_error.sum() / valid_mask.float().sum()
            
            # Accumulate metrics
            epe_sum += epe.item()
            bad1_sum += bad1.item()
            bad2_sum += bad2.item()
            bad3_sum += bad3.item()
            bad5_sum += bad5.item()
            depth_mae_sum += depth_mae.item()
            frame_count += 1
            
            print(f'{dataset}/{keyframe_dir}/{gt_file.name}: EPE={epe.item()}, Bad1={bad1.item()}, Bad2={bad2.item()}, \
                                                             Bad3={bad3.item()}, Bad5={bad5.item()} ,Depth_MAE={depth_mae.item()}')
        
        # Calculate average metrics for this keyframe
        if frame_count > 0:
            keyframe_metrics['dataset'].append(dataset)
            keyframe_metrics['keyframe'].append(keyframe)
            keyframe_metrics['EPE'].append(epe_sum / frame_count)
            keyframe_metrics['Bad1'].append(bad1_sum / frame_count)
            keyframe_metrics['Bad2'].append(bad2_sum / frame_count)
            keyframe_metrics['Bad3'].append(bad3_sum / frame_count)
            keyframe_metrics['Bad5'].append(bad5_sum / frame_count)
            keyframe_metrics['Depth_MAE'].append(depth_mae_sum / frame_count)

# Convert keyframe metrics to DataFrame
df = pd.DataFrame(keyframe_metrics)

# Calculate averages per dataset
dataset_avg = df.groupby('dataset')[['EPE', 'Bad1', 'Bad2', 'Bad3','Bad5', 'Depth_MAE']].mean()
print("\n=== Averages per Dataset ===")
print(dataset_avg)

# Calculate overall averages
overall_avg = df[['EPE', 'Bad1', 'Bad2', 'Bad3','Bad5', 'Depth_MAE']].mean()
print("\n=== Overall Averages ===")
print(overall_avg)

# Save results to CSV
df.to_csv(args.csv, index=False)

# Also save summary statistics
with open(args.csv.replace('.csv', '_summary.csv'), 'w') as f:
    f.write("=== Keyframe Level Results ===\n")
    df.to_csv(f, index=False)
    f.write("\n=== Averages per Dataset ===\n")
    dataset_avg.to_csv(f)
    f.write("\n=== Overall Averages ===\n")
    overall_avg.to_frame().T.to_csv(f)

