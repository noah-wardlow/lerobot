#!/usr/bin/env python3

"""
Simple UR5 camera-only visualization - no Rerun controls, just camera feeds.
"""

import argparse
import logging
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from visualize_ur5_rerun import EpisodeSampler, to_hwc_uint8_numpy


def visualize_ur5_cameras_only(
    dataset: LeRobotDataset,
    episode_index: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
    save_path: str = None,
    hide_panels: bool = False
):
    """Visualize only camera feeds from UR5 dataset."""
    
    repo_id = dataset.repo_id
    logging.info(f"Visualizing cameras for episode {episode_index} from {repo_id}")
    
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
    rr.init(f"UR5_Cameras/episode_{episode_index}", spawn=spawn_viewer)
    
    # Configure UI panels if hiding is requested
    if hide_panels:
        import rerun.blueprint as rrb
        
        # Create a minimal blueprint with just the camera views
        blueprint = rrb.Horizontal(
            rrb.Spatial2DView(name="Main Camera", origin="cameras/image"),
            rrb.Spatial2DView(name="Hand Camera", origin="cameras/hand_image"),
            column_shares=[1, 1]
        )
        
        rr.send_blueprint(blueprint)
    
    # Serve web interface
    if save_path is None:
        rr.serve_web(open_browser=True, web_port=0, ws_port=0)
    
    logging.info("Processing camera frames...")
    
    for batch in tqdm.tqdm(dataloader, desc="Visualizing camera frames"):
        for i in range(len(batch["index"])):
            frame_idx = batch["frame_index"][i].item()
            timestamp = batch["timestamp"][i].item()
            
            # Set timeline
            rr.set_time_sequence("frame", frame_idx)
            rr.set_time_seconds("timestamp", timestamp)
            
            # Log ONLY camera images
            for camera_key in dataset.meta.camera_keys:
                if camera_key in batch:
                    image = to_hwc_uint8_numpy(batch[camera_key][i])
                    # Clean up the camera name for display
                    camera_name = camera_key.replace("observation.images.", "")
                    rr.log(f"cameras/{camera_name}", rr.Image(image))
    
    if save_path:
        rr.save(save_path)
        logging.info(f"Saved to {save_path}")
    
    logging.info("Camera visualization complete!")
    
    # Keep running
    if save_path is None:
        try:
            import signal
            import time
            
            def signal_handler(signum, frame):
                raise KeyboardInterrupt
            
            signal.signal(signal.SIGINT, signal_handler)
            
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Shutting down...")


def main():
    parser = argparse.ArgumentParser(description="UR5 camera-only visualization")
    parser.add_argument("--repo-id", type=str, default="IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hide-panels", action="store_true", help="Hide entity tree and other panels, show only cameras")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        dataset = LeRobotDataset(args.repo_id)
        logging.info(f"Dataset loaded: {dataset.num_episodes} episodes")
        
        if args.episode_index >= dataset.num_episodes:
            raise ValueError(f"Episode {args.episode_index} >= {dataset.num_episodes}")
        
        visualize_ur5_cameras_only(
            dataset=dataset,
            episode_index=args.episode_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_path=args.save,
            hide_panels=args.hide_panels
        )
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())