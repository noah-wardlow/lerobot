#!/usr/bin/env python3

"""
Script to load and explore the Berkeley Autolab UR5 dataset.
Based on the LeRobot dataset loading example.

Usage: PYTHONPATH=src python load_berkeley_ur5.py
"""

from pprint import pprint

print("Attempting to import LeRobot modules...")
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    print("✓ Successfully imported LeRobot modules")
except ImportError as e:
    print(f"✗ Failed to import LeRobot modules: {e}")
    print("Try running: PYTHONPATH=src python load_berkeley_ur5.py")
    exit(1)

# Berkeley Autolab UR5 dataset
repo_id = "IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot"

print(f"Loading dataset: {repo_id}")
print("=" * 50)

try:
    # First, load metadata to understand the dataset structure
    print("Loading dataset metadata...")
    ds_meta = LeRobotDatasetMetadata(repo_id)
    
    print(f"✓ Dataset metadata loaded successfully")
    print(f"Total number of episodes: {ds_meta.total_episodes}")
    print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
    print(f"Frames per second: {ds_meta.fps}")
    print(f"Robot type: {ds_meta.robot_type}")
    print(f"Camera keys: {ds_meta.camera_keys}")
    print()
    
    print("Dataset tasks:")
    pprint(ds_meta.tasks)
    print()
    
    print("Dataset features:")
    pprint(ds_meta.features)
    print()
    
    # Load the actual dataset
    print("Loading full dataset...")
    dataset = LeRobotDataset(repo_id)
    
    print(f"✓ Dataset loaded successfully!")
    print(f"Number of episodes: {dataset.num_episodes}")
    print(f"Number of frames: {dataset.num_frames}")
    print(f"Dataset root: {dataset.root}")
    print()
    
    # Explore a single frame
    print("Exploring first frame...")
    first_frame = dataset[0]
    print("Available keys in first frame:")
    for key in sorted(first_frame.keys()):
        value = first_frame[key]
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    print()
    print("✓ Berkeley Autolab UR5 dataset loaded and ready to use!")
    
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()