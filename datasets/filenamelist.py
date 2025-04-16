import os
import argparse
import glob
import fnmatch

def find_stereo_pairs_in_stereomis(root_path):
    """
    Find stereo image pairs and their corresponding disparity maps in the StereoMIS dataset structure.
    
    StereoMIS dataset has structures like:
    - Left: /workspace/dataset/StereoMIS/P1/left_finalpass/*.png
    - Right: /workspace/dataset/StereoMIS/P1/right_finalpass/*.png
    - Disparity: not available
    """
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Root path not found: {root_path}")
    
    matched_left = []
    matched_right = []
    
    # Find all dataset folders
    for dataset_id in ['P1', 'P2_0', 'P2_1','P2_2','P2_3','P2_4','P2_5','P2_6','P2_7','P2_8']:
        left_dir = os.path.join(root_path, dataset_id, 'left_finalpass')
        right_dir = os.path.join(root_path, dataset_id, 'right_finalpass')
        
        if not all(os.path.exists(d) for d in [left_dir, right_dir]):
            continue
            
        print(f"Processing directory: {dataset_id}")
        
        # Get all image files in the left directory
        for img_file in os.listdir(left_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                left_path = os.path.join(left_dir, img_file)
                right_path = os.path.join(right_dir, img_file)
                
                if os.path.exists(right_path):
                    matched_left.append(left_path)
                    matched_right.append(right_path)
    
    print(f"Successfully matched {len(matched_left)} stereo pairs")
    return matched_left, matched_right


def find_stereo_pairs_in_scared(root_path):
    """
    Find stereo image pairs and their corresponding disparity maps in the SCARED dataset structure.
    
    SCARED dataset has structures like:
    - Left: /path/to/dataset_8/keyframe_0/data/left_rectified/*.png
    - Right: /path/to/dataset_8/keyframe_0/data/right_rectified/*.png
    - Disparity: /path/to/dataset_8/keyframe_0/data/disparity/*.png
    
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
    
    # Find all dataset folders
    for dataset_id in ['dataset_8', 'dataset_9']:
        dataset_path = os.path.join(root_path, dataset_id)
        if not os.path.exists(dataset_path):
            continue
            
        # Process each keyframe
        for keyframe_id in range(5):  # keyframe_0 to keyframe_4
            keyframe_path = os.path.join(dataset_path, f'keyframe_{keyframe_id}', 'data')
            if not os.path.exists(keyframe_path):
                continue
                
            left_dir = os.path.join(keyframe_path, 'left_rectified')
            right_dir = os.path.join(keyframe_path, 'right_rectified')
            disp_dir = os.path.join(keyframe_path, 'disparity')
            
            if not all(os.path.exists(d) for d in [left_dir, right_dir, disp_dir]):
                continue
                
            print(f"Processing directory: {keyframe_path}")
            
            # Get all image files in the left directory
            for img_file in os.listdir(left_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    left_path = os.path.join(left_dir, img_file)
                    right_path = os.path.join(right_dir, img_file)
                    disp_path = os.path.join(disp_dir, img_file)
                    
                    if os.path.exists(right_path) and os.path.exists(disp_path):
                        matched_left.append(left_path)
                        matched_right.append(right_path)
                        matched_disp.append(disp_path)
    
    print(f"Successfully matched {len(matched_left)} stereo pairs with disparity maps")
    return matched_left, matched_right, matched_disp

def find_stereo_pairs_in_ris2017(root_path):
    """
    Find stereo image pairs in the RIS 2017 dataset structure.
    
    RIS 2017 dataset has structures like:
    - Left: /path/to/instrument_dataset_4/left_frame/frame000.png
    - Right: /path/to/instrument_dataset_4/right_frame/frame000.png
    
    Args:
        root_path (str): Root directory to search for images
        
    Returns:
        tuple: Lists of left image paths, right image paths
    """
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Root path not found: {root_path}")
    
    matched_left = []
    matched_right = []
    
    # Find all dataset folders
    dataset_pattern = 'instrument_dataset_*'
    for dataset_dir in glob.glob(os.path.join(root_path, dataset_pattern)):
        left_dir = os.path.join(dataset_dir, 'left_frame')
        right_dir = os.path.join(dataset_dir, 'right_frame')
        
        if not all(os.path.exists(d) for d in [left_dir, right_dir]):
            continue
            
        print(f"Processing directory: {dataset_dir}")
        
        # Get all image files in the left directory
        for img_file in sorted(os.listdir(left_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                left_path = os.path.join(left_dir, img_file)
                right_path = os.path.join(right_dir, img_file)
                
                if os.path.exists(right_path):
                    matched_left.append(left_path)
                    matched_right.append(right_path)
    
    print(f"Successfully matched {len(matched_left)} stereo pairs")
    return matched_left, matched_right

def write_file_paths(left_paths, right_paths, disp_paths, output_file, make_relative=True, base_path=None):
    """
    Write the organized file paths to an output file.
    Paths will be sorted before writing.
    
    Args:
        left_paths (list): List of left image paths
        right_paths (list): List of right image paths
        disp_paths (list): List of disparity paths
        output_file (str): Path to the output file
        make_relative (bool): If True, make paths relative to base_path
        base_path (str): Base path for making relative paths
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Create sorted tuples of paths
    path_tuples = list(zip(left_paths, right_paths, disp_paths))
    path_tuples.sort(key=lambda x: x[0])  # Sort based on left image path
    
    with open(output_file, 'w') as f:
        for left, right, disp in path_tuples:
            if make_relative and base_path:
                left = os.path.relpath(left, base_path)
                right = os.path.relpath(right, base_path)
                disp = os.path.relpath(disp, base_path)
            f.write(f"{left} {right} {disp}\n")
    
    print(f"Successfully wrote {len(path_tuples)} entries to {output_file}")

def write_file_paths_ris2017(left_paths, right_paths, output_file, make_relative=True, base_path=None):
    """
    Write the organized file paths to an output file for RIS 2017 dataset.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Create sorted tuples of paths
    path_tuples = list(zip(left_paths, right_paths))
    path_tuples.sort(key=lambda x: x[0])  # Sort based on left image path
    
    with open(output_file, 'w') as f:
        for left, right in path_tuples:
            if make_relative and base_path:
                left = os.path.relpath(left, base_path)
                right = os.path.relpath(right, base_path)
            f.write(f"{left} {right}\n")
    
    print(f"Successfully wrote {len(path_tuples)} entries to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate stereo dataset file list from RIS 2017 dataset structure.')
    parser.add_argument('--dataset', type=str, choices=['scared', 'ris2017','stereomis'], required=True,
                        help='Dataset type to process')
    parser.add_argument('--root_path', type=str, default='/workspace/dataset/ris_2017/test_set',
                        help='Root directory containing the dataset')
    parser.add_argument('--output_file', type=str, default='./ris2017_testset.txt',
                        help='Path to output file where paths will be written')
    parser.add_argument('--relative', action='store_true', default=False,
                        help='Make output paths relative to root_path')
    
    args = parser.parse_args()
    
    if args.dataset == 'scared':
        left_paths, right_paths, disp_paths = find_stereo_pairs_in_scared(args.root_path)
        write_file_paths(
            left_paths, 
            right_paths, 
            disp_paths, 
            args.output_file,
            make_relative=args.relative,
            base_path=args.root_path if args.relative else None
        )
    elif args.dataset == 'ris2017':
        left_paths, right_paths = find_stereo_pairs_in_ris2017(args.root_path)
        write_file_paths_ris2017(
            left_paths, 
            right_paths, 
            args.output_file,
            make_relative=args.relative,
            base_path=args.root_path if args.relative else None
        )
    else:  # stereomis
        left_paths, right_paths = find_stereo_pairs_in_stereomis(args.root_path)
        write_file_paths_ris2017(
            left_paths, 
            right_paths, 
            args.output_file,
            make_relative=args.relative,
            base_path=args.root_path if args.relative else None
        )

if __name__ == "__main__":
    main()