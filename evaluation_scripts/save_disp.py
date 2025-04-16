import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import os
import skimage.io
from utils import Params
from models.StereoMamba import StereoMamba
from torchvision import transforms
from datasets.data_io import read_all_lines
import time
import matplotlib.pyplot as plt

DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_transform():
    if args.images_filename.endswith('sceneflow_testing.txt'):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
    elif args.images_filename.endswith('scared_testset.txt'):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5339, 0.3666, 0.4537],
                            std=[0.2006, 0.1978, 0.2128])
        ])
    elif args.images_filename.endswith('ris2017_trainset.txt') or args.images_filename.endswith('ris2017_testset.txt'):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4899, 0.3078, 0.3474],
                            std=[0.1800, 0.1592, 0.1763])
        ])
    elif args.images_filename.endswith('stereomis_testing.txt'):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2516, 0.3255, 0.5639],
                            std=[0.2002, 0.2060, 0.2205])
        ])

def load_image(imfile):
    img = Image.open(imfile).convert('RGB')
    w, h = img.size
    # import ipdb; ipdb.set_trace()
    processed = get_transform()
    img = processed(img).numpy()
    
    # Pad to target size
    if args.images_filename.endswith('sceneflow_testing.txt'):
        target_height = 540
        target_width = 960
    elif args.images_filename.endswith('scared_testset.txt'):
        target_height = 1024
        target_width = 1280
    elif args.images_filename.endswith('ris2017_trainset.txt') or args.images_filename.endswith('ris2017_testset.txt'):
        target_height = 1024
        target_width = 1280
    elif args.images_filename.endswith('stereomis_testing.txt'):
        target_height = 1024
        target_width = 1280
    
    top_pad = max(target_height - h, 0)
    right_pad = max(target_width - w, 0)
    
    img = np.lib.pad(img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE), (h, w), top_pad, right_pad

def load_paths(list_filename):
    lines = read_all_lines(list_filename)
    splits = [line.split() for line in lines]
    left_images = [x[0] for x in splits]
    right_images = [x[1] for x in splits]
    return left_images, right_images

def demo(args):
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
    model.to(DEVICE)
    model.eval()
    
    model.load_state_dict(torch.load(args.restore_ckpt))
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        left_images, right_images = load_paths(args.images_filename)
        print(f"Found {len(left_images)} image pairs. Saving files to {output_directory}/")

        for (left_name, right_name) in tqdm(list(zip(left_images, right_images))):
            # Construct full paths
            imfile1 = os.path.join(os.path.dirname(args.images_filename), '..', left_name)
            imfile2 = os.path.join(os.path.dirname(args.images_filename), '..', right_name)
            
            image1, (h, w), top_pad, right_pad = load_image(imfile1)
            image2, _, _, _ = load_image(imfile2)
            
            start_time = time.time()
            
            disparity_predictions = model(image1, image2)
            end_time = time.time()
            print(f"Time taken for inference: {end_time - start_time}")
            disp = disparity_predictions[-1]
            import ipdb; ipdb.set_trace()
            # Remove padding
            if top_pad > 0:
                disp = disp[:, top_pad:, :]
            if right_pad > 0:
                disp = disp[:, :, :-right_pad]
            disp = disp[:, :h, :w]
            
            # Save with original filename structure
            if args.images_filename.endswith('scared_testset.txt'):
                out_filename = os.path.join(output_directory, 
                                      left_name.split('/')[5],
                                      left_name.split('/')[6], 
                                      left_name.split('/')[-1])
            elif args.images_filename.endswith('ris2017_trainset.txt') or args.images_filename.endswith('ris2017_testset.txt'):
                out_filename = os.path.join(output_directory, 
                                      left_name.split('/')[4],
                                      left_name.split('/')[5], 
                                      left_name.split('/')[-1])
            elif args.images_filename.endswith('stereomis_testing.txt'):
                # import ipdb; ipdb.set_trace()
                out_filename = os.path.join(output_directory, 
                                      left_name.split('/')[4],
                                      left_name.split('/')[-1])
            # out_filename = os.path.join(output_directory, 
            #                           left_name.split('/')[5],
            #                           left_name.split('/')[6], 
            #                           left_name.split('/')[-1])
            os.makedirs(os.path.dirname(out_filename), exist_ok=True)
            # import ipdb; ipdb.set_trace()
            disp = disp.cpu().numpy().squeeze()
            skimage.io.imsave(out_filename, np.round(disp * 128).astype(np.uint16))
            # plt.imsave(out_filename.replace('.png', '_color.png'), disp, cmap='jet') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='/workspace/StereoMamba2/checkpoints/Finetune_on_scared_cross_attn_lr1_start_from_epoch_25/Finetune_on_scared_epoch_100_error_1.513.pt')
    parser.add_argument('--default_config', help="config file", default='/workspace/StereoMamba2/training_configs/fine_tune/config_scared.json')
    parser.add_argument('--output_directory', help="directory to save output", default='/workspace/StereoMamba2/detele_me')
    parser.add_argument('--images_filename', help="path to image list file", default='/workspace/StereoMamba2/datasets/filenames/ris2017_testset.txt')
    
    args = parser.parse_args()
    config = Params(args.default_config)
    demo(args)