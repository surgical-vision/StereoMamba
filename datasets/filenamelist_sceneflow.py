import os
import argparse
import glob
import fnmatch

def find_stereo_pairs_in_sceneflow(root_path):
    """
    Find stereo image pairs and their corresponding disparity maps in the SceneFlow dataset structure,
    which may have multiple levels of nested directories.
    
    SceneFlow dataset typically has structures like:
    - Left: /path/to/frames_finalpass/TRAIN/15mm_focallength/scene_backwards/fast/left/*.png
    - Right: /path/to/frames_finalpass/TRAIN/15mm_focallength/scene_backwards/fast/right/*.png
    - Disparity: /path/to/disparity/TRAIN/15mm_focallength/scene_backwards/fast/left/*.pfm
    
    Args:
        root_path (str): Root directory to search for images
        
    Returns:
        tuple: Lists of left image paths, right image paths, and disparity paths
    """
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Root path not found: {root_path}")
    
    matched_left = []
    matched_right = []
    matched_disp = []
    
    # Find all 'left' directories in frames_finalpass or frames_cleanpass
    left_dirs = []
    for frames_type in ['frames_finalpass', 'frames_cleanpass']:
        frames_path = os.path.join(root_path, frames_type)
        if os.path.exists(frames_path):
            for root, dirs, files in os.walk(frames_path):
                if os.path.basename(root) == 'left':
                    left_dirs.append(root)
    
    print(f"Found {len(left_dirs)} 'left' directories")
    
    # Process each left directory
    for left_dir in left_dirs:
        # Get path components to construct the right and disparity paths
        dir_parts = left_dir.split(os.path.sep)
        frames_type_index = -1
        
        # Find the index of frames_finalpass or frames_cleanpass
        for i, part in enumerate(dir_parts):
            if part.startswith('frames_'):
                frames_type_index = i
                break
        
        if frames_type_index == -1:
            continue
        
        # Get the path after frames_finalpass/frames_cleanpass
        subpath = os.path.sep.join(dir_parts[frames_type_index + 1:-1])  # Exclude 'left'
        
        # Construct right directory path
        right_dir = os.path.join(root_path, dir_parts[frames_type_index], subpath, 'right')
        
        # Construct disparity directory path - replace 'frames_*' with 'disparity'
        disp_dir = os.path.join(root_path, 'disparity', subpath, 'left')
        
        # Check if both directories exist
        if not (os.path.exists(right_dir) and os.path.exists(disp_dir)):
            print(f"Skipping {left_dir} - matching directories not found")
            continue
        
        print(f"Processing directory: {left_dir}")
        
        # Get all image files in the left directory
        left_images = []
        for img_file in os.listdir(left_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                left_images.append(os.path.join(left_dir, img_file))
        
        # Process each left image
        for left_path in left_images:
            # Extract just the filename
            filename = os.path.basename(left_path)
            filename_no_ext = os.path.splitext(filename)[0]
            
            # Construct the expected right image path
            right_path = os.path.join(right_dir, filename)
            
            # Construct the expected disparity image path (pfm extension)
            disp_path = os.path.join(disp_dir, f"{filename_no_ext}.pfm")
            
            # If both right and disparity images exist, add them to the lists
            if os.path.exists(right_path) and os.path.exists(disp_path):
                matched_left.append(left_path)
                matched_right.append(right_path)
                matched_disp.append(disp_path)
            else:
                if not os.path.exists(right_path):
                    print(f"Right image not found: {right_path}")
                if not os.path.exists(disp_path):
                    print(f"Disparity image not found: {disp_path}")
    
    print(f"Successfully matched {len(matched_left)} stereo pairs with disparity maps")
    
    return matched_left, matched_right, matched_disp

def write_file_paths(left_paths, right_paths, disp_paths, output_file, make_relative=True, base_path=None):
    """
    Write the organized file paths to an output file.
    
    Args:
        left_paths (list): List of left image paths
        right_paths (list): List of right image paths
        disp_paths (list): List of disparity paths
        output_file (str): Path to the output file
        make_relative (bool): If True, make paths relative to base_path
        base_path (str): Base path for making relative paths
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for left, right, disp in zip(left_paths, right_paths, disp_paths):
            if make_relative and base_path:
                left = os.path.relpath(left, base_path)
                right = os.path.relpath(right, base_path)
                disp = os.path.relpath(disp, base_path)
            f.write(f"{left} {right} {disp}\n")
    
    print(f"Successfully wrote {len(left_paths)} entries to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate stereo dataset file list from SceneFlow structure.')
    parser.add_argument('--root_path', type=str, default='/workspace/dataset/scared/test_data',
                        help='Root directory containing the SceneFlow dataset')
    parser.add_argument('--output_file', type=str, default='/workspace/StereoMamba2/datasets/filenames/flyingthings_pairs.txt',
                        help='Path to output file where paths will be written')
    parser.add_argument('--relative', action='store_true', default=False,
                        help='Make output paths relative to root_path')
    
    args = parser.parse_args()
    
    left_paths, right_paths, disp_paths = find_stereo_pairs_in_sceneflow(args.root_path)
    
    write_file_paths(
        left_paths, 
        right_paths, 
        disp_paths, 
        args.output_file,
        make_relative=args.relative,
        base_path=args.root_path if args.relative else None
    )

if __name__ == "__main__":
    main()