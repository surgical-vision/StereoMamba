import argparse
from pathlib import Path
import torch
import json
from utils.error_metrics import disparity_bad3, disparity_epe
import numpy as np
import pandas as pd
from PIL import Image
import os
from models.StereoMamba import StereoMamba
from utils import Params
from datasets import sceneflow_dataset
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
parser.add_argument('--restoreckpt', default='/workspace/StereoMamba2/checkpoints/pretrain_cross_attn/Pretrain_on_sceneflow_lowest_eval_loss.pt')
parser.add_argument('--default_config', default='/workspace/StereoMamba2/training_configs/pretrain/config_sceneflow.json')
parser.add_argument('--csv', default='./stereomamba_sceneflow_results.csv')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize keyframe-level metrics storage
keyframe_metrics = {
    'dataset': [], 'keyframe': [],
    'EPE': [], 'Bad3': [], 'Depth_MAE': []
}

# main
if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    config = Params(args.default_config)
    # Load the model
    model = StereoMamba(patch_size=config.patch_size, in_chans=config.in_chans, depths=config.depths, 
                    dims=config.dims, ssm_d_state=config.ssm_d_state, ssm_ratio=config.ssm_ratio,
                    ssm_dt_rank=config.ssm_dt_rank, ssm_act_layer=config.ssm_act_layer, ssm_conv=config.ssm_conv,
                    ssm_conv_bias=config.ssm_conv_bias, ssm_drop_rate=config.ssm_drop_rate, ssm_init=config.ssm_init,
                    forward_type=config.forward_type, mlp_ratio=config.mlp_ratio, mlp_act_layer=config.mlp_act_layer,
                    mlp_drop_rate=config.mlp_drop_rate, gmlp=config.gmlp, drop_path_rate=config.drop_path_rate,
                    patch_norm=config.patch_norm, norm_layer=config.norm_layer, downsample_version=config.downsample_version,
                    patchembed_version=config.patchembed_version, use_checkpoint=config.use_checkpoint, posembed=config.posembed,
                    imgsize=config.imgsize, max_disparity=config.max_disparity, use_concat_volume=config.use_concat_volume,
                    cross_attn=config.cross_attn, d_model=config.d_model, d_state=config.d_state, d_conv=config.d_conv,
                    expand=config.expand, headdim=config.headdim, ngroups=config.ngroups)
    model.load_state_dict(torch.load(args.restoreckpt))
    model.to(device)
    model.eval()

    # Create SceneFlow test dataset
    test_dataset = sceneflow_dataset.SceneFlowDatset(
        datapath="/workspace/dataset/sceneflow",
        list_filename="/workspace/StereoMamba2/datasets/filenames/sceneflow_testing.txt",
        training=False
    )

    # Initialize metrics
    all_epe = []
    all_bad1 = []
    all_bad2 = []
    all_bad3 = []
    all_bad5 = []
    batch_results = {
        'batch': [],
        'avg_epe': [],
        'avg_bad1': [],
        'avg_bad2': [],
        'avg_bad3': [],
        'avg_bad5': []
    }

    with torch.no_grad():
        for i in range(len(test_dataset)):
            # Load sample
            sample = test_dataset[i]
            left = sample['left'].to(device).unsqueeze(0)
            right = sample['right'].to(device).unsqueeze(0)
            gt_disp = torch.from_numpy(sample['disparity']).to(device).unsqueeze(0)

            # Model prediction
            # start_time = time.time()
            pred_disp = model(left, right)
            # end_time = time.time()
            # print(f"Image {i+1}: Inference Time: {end_time - start_time:.3f} seconds")

            # Calculate metrics
            epe = disparity_epe(pred_disp[-1], gt_disp, max_disparity=192)
            bad1 = disparity_bad3(pred_disp[-1], gt_disp, max_disparity=192, thres=1)
            bad2 = disparity_bad3(pred_disp[-1], gt_disp, max_disparity=192, thres=2)
            bad3 = disparity_bad3(pred_disp[-1], gt_disp, max_disparity=192, thres=3)
            bad5 = disparity_bad3(pred_disp[-1], gt_disp, max_disparity=192, thres=5)
            # print(f"Image {i+1}: EPE: {epe.item():.3f}, Bad3: {bad3.item():.3f}%")
            # Store metrics
            all_epe.append(epe.item())
            all_bad1.append(bad1.item())
            all_bad2.append(bad2.item())
            all_bad3.append(bad3.item())
            all_bad5.append(bad5.item())

            if (i + 1) % 100 == 0:
                batch_avg_epe = np.mean(all_epe[-100:])
                batch_avg_bad1 = np.mean(all_bad1[-100:])
                batch_avg_bad2 = np.mean(all_bad2[-100:])
                batch_avg_bad3 = np.mean(all_bad3[-100:])
                batch_avg_bad5 = np.mean(all_bad5[-100:])
                batch_results['batch'].append(i+1)
                batch_results['avg_epe'].append(batch_avg_epe)
                batch_results['avg_bad1'].append(batch_avg_bad1)
                batch_results['avg_bad2'].append(batch_avg_bad2)
                batch_results['avg_bad3'].append(batch_avg_bad3)
                batch_results['avg_bad5'].append(batch_avg_bad5)
                print(f"Batch {(i+1)//100} (Images {i-98}-{i+1})")
                print(f"Batch Average EPE: {batch_avg_epe:.3f}")
                print(f"Batch Average Bad1: {batch_avg_bad1:.3f}")
                print(f"Batch Average Bad2: {batch_avg_bad2:.3f}")
                print(f"Batch Average Bad3: {batch_avg_bad3:.3f}")
                print(f"Batch Average Bad5: {batch_avg_bad5:.3f}")
                print(f"Processed {i+1}/{len(test_dataset)} images")

    # Calculate average metrics
    avg_epe = np.mean(all_epe)
    avg_bad1 = np.mean(all_bad1)
    avg_bad2 = np.mean(all_bad2)
    avg_bad3 = np.mean(all_bad3)
    avg_bad5 = np.mean(all_bad5)
    print(f"Evaluation completed!")
    print(f"Average EPE: {avg_epe:.3f}")
    print(f"Average Bad1: {avg_bad1:.3f}")
    print(f"Average Bad2: {avg_bad2:.3f}")
    print(f"Average Bad3: {avg_bad3:.3f}")
    print(f"Average Bad5: {avg_bad5:.3f}")
    # Save both overall and batch results
    results = {
        'EPE': [avg_epe],
        'Bad1': [avg_bad1],
        'Bad2': [avg_bad2],
        'Bad3': [avg_bad3],
        'Bad5': [avg_bad5]
    }
    df = pd.DataFrame(results)
    df.to_csv(args.csv, index=False)

    # Save batch results to a separate CSV
    batch_df = pd.DataFrame(batch_results)
    batch_csv = args.csv.replace('.csv', '_batch_results.csv')
    batch_df.to_csv(batch_csv, index=False)
    

