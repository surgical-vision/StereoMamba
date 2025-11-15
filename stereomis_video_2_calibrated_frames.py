import cv2
import os
import argparse
from pathlib import Path
import shutil
import numpy as np
import configparser

def read_calibration(calib_file):
    """
    Read calibration parameters from StereoCalibration.ini file.
    
    Args:
        calib_file: Path to calibration file
    
    Returns:
        Dictionary containing camera matrices and distortion coefficients
    """
    config = configparser.ConfigParser()
    config.read(calib_file)
    
    # Left camera parameters
    left_K = np.array([
        [float(config['StereoLeft']['fc_x']), 0, float(config['StereoLeft']['cc_x'])],
        [0, float(config['StereoLeft']['fc_y']), float(config['StereoLeft']['cc_y'])],
        [0, 0, 1]
    ])
    
    left_dist = np.array([
        float(config['StereoLeft'][f'kc_{i}']) for i in range(8)
    ])
    
    # Right camera parameters
    right_K = np.array([
        [float(config['StereoRight']['fc_x']), 0, float(config['StereoRight']['cc_x'])],
        [0, float(config['StereoRight']['fc_y']), float(config['StereoRight']['cc_y'])],
        [0, 0, 1]
    ])
    
    right_dist = np.array([
        float(config['StereoRight'][f'kc_{i}']) for i in range(8)
    ])
    
    # Rotation and translation from left to right camera
    R = np.array([
        [float(config['StereoRight'][f'R_{i}']) for i in range(3)],
        [float(config['StereoRight'][f'R_{i}']) for i in range(3, 6)],
        [float(config['StereoRight'][f'R_{i}']) for i in range(6, 9)]
    ])
    
    T = np.array([
        float(config['StereoRight']['T_0']),
        float(config['StereoRight']['T_1']),
        float(config['StereoRight']['T_2'])
    ])
    
    image_size = (
        int(config['StereoLeft']['res_x']),
        int(config['StereoLeft']['res_y'])
    )
    
    return {
        'left_K': left_K,
        'left_dist': left_dist,
        'right_K': right_K,
        'right_dist': right_dist,
        'R': R,
        'T': T,
        'image_size': image_size
    }

def compute_rectification_maps(calib_params):
    """
    Compute rectification maps for stereo images.
    
    Args:
        calib_params: Dictionary containing calibration parameters
    
    Returns:
        Rectification maps for left and right images
    """
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        calib_params['left_K'],
        calib_params['left_dist'][:5],  # Use only first 5 distortion coefficients
        calib_params['right_K'],
        calib_params['right_dist'][:5],
        calib_params['image_size'],
        calib_params['R'],
        calib_params['T'],
        alpha=-1  # Changed from 0 to 1 to retain all pixels and avoid cropping
    )
    
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        calib_params['left_K'],
        calib_params['left_dist'][:5],
        R1,
        P1,
        calib_params['image_size'],
        cv2.CV_32FC1
    )
    
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        calib_params['right_K'],
        calib_params['right_dist'][:5],
        R2,
        P2,
        calib_params['image_size'],
        cv2.CV_32FC1
    )
    
    return {
        'map1_left': map1_left,
        'map2_left': map2_left,
        'map1_right': map1_right,
        'map2_right': map2_right
    }

def process_video_and_extract(video_path, input_folder, output_folder, frame_interval=8):
    """
    Process video: split to stereo, calibrate, rectify, and extract masks/poses at intervals.
    
    Args:
        video_path: Path to input video
        input_folder: Path to folder containing masks and groundtruth.txt
        output_folder: Path to save output
        frame_interval: Extract every Nth frame
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output directories
    frames_left_dir = output_path / "frames" / "left"
    frames_right_dir = output_path / "frames" / "right"
    frames_left_rect_dir = output_path / "frames_rectified" / "left"
    frames_right_rect_dir = output_path / "frames_rectified" / "right"
    masks_left_dir = output_path / "masks"
    
    frames_left_dir.mkdir(parents=True, exist_ok=True)
    frames_right_dir.mkdir(parents=True, exist_ok=True)
    frames_left_rect_dir.mkdir(parents=True, exist_ok=True)
    frames_right_rect_dir.mkdir(parents=True, exist_ok=True)
    masks_left_dir.mkdir(parents=True, exist_ok=True)
    
    # Read calibration and compute rectification maps
    calib_file = input_path / "StereoCalibration.ini"
    if not calib_file.exists():
        print("Error: Calibration file not found!")
        return
    
    print("Reading calibration file...")
    calib_params = read_calibration(calib_file)
    rect_maps = compute_rectification_maps(calib_params)
    print("Rectification maps computed successfully")
    
    # Read all poses from groundtruth.txt
    groundtruth_file = input_path / "groundtruth.txt"
    poses = {}
    if groundtruth_file.exists():
        with open(groundtruth_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    frame_id = int(parts[0])
                    pose_data = ' '.join(parts[1:])
                    poses[frame_id] = pose_data
    
    masks_input = input_path / "masks"
    
    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    extracted_poses = []
    extracted_count = 0
    frame_count = 0
    
    print("Processing video frames...")
    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            break
        
        # Check frame dimensions
        if frame.shape[0] < 2048:
            print("Error: Frame height is less than expected for splitting.")
            break
        
        # Process every frame_interval frames
        if frame_count % frame_interval == 0:
            # Split frame to left and right
            left_img = frame[:1024]
            right_img = frame[1024:]
            
            # Save original frames
            cv2.imwrite(str(frames_left_dir / f"{extracted_count:06d}.png"), left_img)
            cv2.imwrite(str(frames_right_dir / f"{extracted_count:06d}.png"), right_img)
            
            # Rectify frames
            left_rect = cv2.remap(left_img, rect_maps['map1_left'], 
                                rect_maps['map2_left'], cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, rect_maps['map1_right'], 
                                  rect_maps['map2_right'], cv2.INTER_LINEAR)
            
            cv2.imwrite(str(frames_left_rect_dir / f"{extracted_count:06d}.png"), left_rect)
            cv2.imwrite(str(frames_right_rect_dir / f"{extracted_count:06d}.png"), right_rect)
            
            # Copy mask if exists (masks are named by original frame number)
            left_mask = masks_input / f"{frame_count:06d}l.png"
            if left_mask.exists():
                shutil.copy2(left_mask, masks_left_dir / f"{extracted_count:06d}.png")
            
            # Extract pose (poses are indexed by original frame number)
            if frame_count in poses:
                extracted_poses.append(f"{extracted_count} {poses[frame_count]}")
            
            extracted_count += 1
        
        frame_count += 1
    
    video_capture.release()
    
    # Save extracted poses
    output_groundtruth = output_path / "groundtruth.txt"
    with open(output_groundtruth, 'w') as f:
        f.write('\n'.join(extracted_poses))
    
    print(f"Processing complete!")
    print(f"Total video frames: {frame_count}")
    print(f"Extracted {extracted_count} frames, masks, and poses at interval {frame_interval}")
    print(f"Output saved to: {output_path}")

# Usage
if __name__ == "__main__":
    video_path = '/workspace/dataset/StereoMIS/P2_6/IFBS_ENDOSCOPE-part0006.mp4'
    input_folder = '/workspace/dataset/StereoMIS/P2_6'
    output_folder = '/workspace/scared_toolkit/stereomis/p2_6'
    
    process_video_and_extract(video_path, input_folder, output_folder, frame_interval=2)
