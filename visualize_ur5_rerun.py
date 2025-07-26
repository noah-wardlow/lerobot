#!/usr/bin/env python3

"""
Comprehensive UR5 data visualization with Rerun.
This script visualizes Berkeley Autolab UR5 dataset with:
- Robot joint trajectories in 3D
- Camera streams (if available)
- Action and state time series
- Gripper control visualization
- End-effector path in 3D space

Usage:
    python visualize_ur5_rerun.py [--episode-index 0] [--batch-size 32]
    python visualize_ur5_rerun.py --episode-index 1
"""

import argparse
import logging
import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    """Sampler to iterate through frames of a specific episode."""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 tensor to HWC uint8 numpy array for Rerun."""
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but got {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def compute_ur5_forward_kinematics(joint_angles):
    """
    Simplified UR5 forward kinematics to compute end-effector position.
    This is a basic approximation for visualization purposes.
    
    Args:
        joint_angles: Joint angles in radians (6,)
    
    Returns:
        end_effector_pos: 3D position of end-effector (3,)
    """
    # UR5 DH parameters (approximate)
    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]  # d parameters
    a = [0, -0.42500, -0.39225, 0, 0, 0]  # a parameters
    alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]  # alpha parameters
    
    # Simple forward kinematics computation
    x, y, z = 0, 0, d[0]
    
    # Simplified computation focusing on major link contributions
    if len(joint_angles) >= 6:
        q1, q2, q3, q4, q5, q6 = joint_angles[:6]
        
        # Base rotation
        x += a[1] * np.cos(q1) * np.cos(q2)
        y += a[1] * np.sin(q1) * np.cos(q2)
        z += a[1] * np.sin(q2)
        
        # Second link
        x += a[2] * np.cos(q1) * np.cos(q2 + q3)
        y += a[2] * np.sin(q1) * np.cos(q2 + q3)
        z += a[2] * np.sin(q2 + q3)
        
        # Add offset for remaining links
        x += d[3] * np.cos(q1) * np.sin(q2 + q3)
        y += d[3] * np.sin(q1) * np.sin(q2 + q3)
        z += d[3] * np.cos(q2 + q3)
    
    return np.array([x, y, z])


def visualize_ur5_dataset(
    dataset: LeRobotDataset,
    episode_index: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
    save_path: str = None
):
    """
    Visualize UR5 dataset episode with Rerun.
    
    Args:
        dataset: LeRobotDataset instance
        episode_index: Episode to visualize
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        save_path: Optional path to save .rrd file
    """
    repo_id = dataset.repo_id
    
    logging.info(f"Visualizing episode {episode_index} from {repo_id}")
    
    # Setup data loader
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )
    
    # Initialize Rerun
    spawn_viewer = save_path is None
    rr.init(f"UR5_Visualization/episode_{episode_index}", spawn=spawn_viewer)
    
    # Also serve web interface
    if save_path is None:
        rr.serve_web_viewer(open_browser=True, web_port=0)
    
    # Set up the scene structure
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    
    # Robot base coordinate frame
    rr.log("world/robot_base", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    
    # Store trajectory for path visualization
    end_effector_positions = []
    
    logging.info("Processing frames...")
    
    for batch in tqdm.tqdm(dataloader, desc="Visualizing frames"):
        # Process each frame in the batch
        for i in range(len(batch["index"])):
            frame_idx = batch["frame_index"][i].item()
            timestamp = batch["timestamp"][i].item()
            
            # Set timeline
            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("timestamp", timestamp=timestamp)
            
            # Log camera images if available
            for camera_key in dataset.meta.camera_keys:
                if camera_key in batch:
                    image = to_hwc_uint8_numpy(batch[camera_key][i])
                    rr.log(f"cameras/{camera_key}", rr.Image(image))
            
            # Log robot state
            if "observation.state" in batch:
                joint_positions = batch["observation.state"][i].numpy()
                
                # Log individual joint values
                for joint_idx, joint_pos in enumerate(joint_positions):
                    rr.log(f"robot_state/joint_{joint_idx}", rr.Scalars(joint_pos))
                
                # Compute and visualize end-effector position
                if len(joint_positions) >= 6:
                    ee_pos = compute_ur5_forward_kinematics(joint_positions)
                    end_effector_positions.append(ee_pos)
                    
                    # Log end-effector position
                    rr.log("world/robot_base/end_effector", rr.Points3D([ee_pos], radii=0.02, colors=[1.0, 0.0, 0.0]))
                    
                    # Log end-effector trajectory
                    if len(end_effector_positions) > 1:
                        trajectory = np.array(end_effector_positions)
                        rr.log(
                            "world/robot_base/trajectory",
                            rr.LineStrips3D([trajectory], colors=[0.0, 1.0, 0.0], radii=0.005)
                        )
            
            # Log action commands
            if "action" in batch:
                action = batch["action"][i].numpy()
                
                # Log action components
                action_labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
                for action_idx, (action_val, label) in enumerate(zip(action, action_labels)):
                    if action_idx < len(action_labels):
                        rr.log(f"actions/{label}", rr.Scalars(action_val))
                    else:
                        rr.log(f"actions/dim_{action_idx}", rr.Scalars(action_val))
                
                # Visualize action as arrow in 3D (translation component)
                if len(action) >= 3:
                    action_origin = end_effector_positions[-1] if end_effector_positions else np.zeros(3)
                    action_vector = action[:3] * 0.1  # Scale for visualization
                    action_end = action_origin + action_vector
                    
                    rr.log(
                        "world/robot_base/action_vector",
                        rr.Arrows3D(
                            origins=[action_origin],
                            vectors=[action_vector],
                            colors=[0.0, 0.0, 1.0]
                        )
                    )
            
            # Log gripper state if available
            if "observation.state" in batch:
                state = batch["observation.state"][i].numpy()
                if len(state) > 6:  # Assuming gripper state is last element
                    gripper_state = state[-1]
                    rr.log("robot_state/gripper", rr.Scalars(gripper_state))
            
            # Log episode metadata
            rr.log("episode_info/frame_index", rr.Scalars(frame_idx))
            rr.log("episode_info/timestamp", rr.Scalars(timestamp))
            
            # Log any additional fields
            for key in batch:
                if key not in ["index", "frame_index", "timestamp", "observation.state", "action"] and not key.startswith("observation.images"):
                    value = batch[key][i]
                    if hasattr(value, 'numel') and value.numel() == 1:  # Scalar tensor values
                        rr.log(f"metadata/{key}", rr.Scalars(value.item()))
                    elif isinstance(value, (int, float)):  # Direct scalar values
                        rr.log(f"metadata/{key}", rr.Scalars(value))
                    # Skip string values and complex tensors
    
    # Log final trajectory statistics
    if end_effector_positions:
        trajectory = np.array(end_effector_positions)
        rr.log("trajectory_stats/total_distance", rr.Scalars(
            np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        ))
        rr.log("trajectory_stats/max_position", rr.Scalars(np.max(trajectory)))
        rr.log("trajectory_stats/min_position", rr.Scalars(np.min(trajectory)))
    
    # Save if requested
    if save_path:
        rr.save(save_path)
        logging.info(f"Saved visualization to {save_path}")
    
    logging.info("Visualization complete!")
    
    # Keep server running for web interface
    if save_path is None:
        logging.info("Server running - check terminal output for actual URLs")
        logging.info("Press Ctrl+C to stop...")
        try:
            import signal
            import time
            
            # Set up signal handler for graceful shutdown
            def signal_handler(signum, frame):
                logging.info("Received interrupt signal, shutting down server...")
                raise KeyboardInterrupt
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            while True:
                time.sleep(0.1)  # Shorter sleep for more responsive interruption
        except KeyboardInterrupt:
            logging.info("Shutting down server...")
            return


def main():
    parser = argparse.ArgumentParser(description="Visualize UR5 dataset with Rerun")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot",
        help="Repository ID for the UR5 dataset"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to visualize"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loading"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save .rrd file (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load dataset
        logging.info(f"Loading dataset {args.repo_id}")
        dataset = LeRobotDataset(args.repo_id)
        
        logging.info(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
        
        # Check episode bounds
        if args.episode_index >= dataset.num_episodes:
            raise ValueError(f"Episode index {args.episode_index} >= num_episodes {dataset.num_episodes}")
        
        # Visualize
        visualize_ur5_dataset(
            dataset=dataset,
            episode_index=args.episode_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_path=args.save
        )
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())