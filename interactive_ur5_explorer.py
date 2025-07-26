#!/usr/bin/env python3

"""
Interactive CLI for exploring Berkeley Autolab UR5 dataset.
Allows users to select tasks and episodes interactively, then visualizes them.

Usage: PYTHONPATH=src python interactive_ur5_explorer.py
"""

import logging
import sys
from typing import Dict, List

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from visualize_ur5_rerun import visualize_ur5_dataset
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Try running: PYTHONPATH=src python interactive_ur5_explorer.py")
    sys.exit(1)


def display_tasks(tasks: Dict[int, str]) -> None:
    """Display available tasks in a formatted way."""
    print("\n" + "="*60)
    print("Available Tasks:")
    print("="*60)
    for task_id, task_description in tasks.items():
        print(f"  {task_id}: {task_description}")
    print("="*60)


def get_episodes_for_task(dataset: LeRobotDataset, task_id: int) -> List[int]:
    """Get all episode indices that belong to a specific task."""
    episodes = []
    target_task_description = dataset.meta.tasks[task_id]
    
    for episode_idx in range(dataset.num_episodes):
        # Get the first frame of the episode to check task
        frame_start = dataset.episode_data_index["from"][episode_idx].item()
        frame_data = dataset[frame_start]
        
        # Check both task_index and task string to be sure
        frame_task_index = frame_data["task_index"].item()
        frame_task_string = frame_data["task"]
        
        if frame_task_index == task_id and frame_task_string == target_task_description:
            episodes.append(episode_idx)
    
    return episodes


def get_user_choice(prompt: str, valid_choices: List[int], allow_quit: bool = True) -> int:
    """Get user input with validation."""
    while True:
        try:
            if allow_quit:
                print(f"\n{prompt} (or 'q' to quit):")
            else:
                print(f"\n{prompt}:")
            
            choice = input("> ").strip()
            
            if allow_quit and choice.lower() in ['q', 'quit', 'exit']:
                print("Goodbye!")
                sys.exit(0)
            
            choice_int = int(choice)
            if choice_int in valid_choices:
                return choice_int
            else:
                print(f"Invalid choice. Please select from: {valid_choices}")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)


def main():
    print("🤖 Interactive UR5 Dataset Explorer")
    print("Loading Berkeley Autolab UR5 dataset...")
    
    try:
        # Load dataset
        repo_id = "IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot"
        dataset = LeRobotDataset(repo_id)
        
        print(f"✓ Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
        
        # Get task information
        tasks = dataset.meta.tasks
        
        while True:
            # Display tasks
            display_tasks(tasks)
            
            # Get task selection
            valid_task_ids = list(tasks.keys())
            selected_task_id = get_user_choice(
                f"Select a task ({', '.join(map(str, valid_task_ids))})", 
                valid_task_ids
            )
            
            # Get episodes for selected task
            print(f"\nFinding episodes for task {selected_task_id}: '{tasks[selected_task_id]}'...")
            episodes_for_task = get_episodes_for_task(dataset, selected_task_id)
            
            if not episodes_for_task:
                print(f"No episodes found for task {selected_task_id}!")
                continue
            
            print(f"Found {len(episodes_for_task)} episodes for this task")
            print(f"Episode indices: {episodes_for_task}")
            
            # Add warning for task 4 specifically
            if selected_task_id == 4:
                print("⚠️  WARNING: Task 4 episodes (499, 566) appear to be mislabeled in this dataset.")
                print("   They may show 'blue cup' task content instead of 'marker' task content.")
            
            # Get episode selection
            selected_episode = get_user_choice(
                f"Select an episode {episodes_for_task}", 
                episodes_for_task,
                allow_quit=False
            )
            
            # Get first frame to show episode info
            frame_start = dataset.episode_data_index["from"][selected_episode].item()
            frame_end = dataset.episode_data_index["to"][selected_episode].item()
            frame_count = frame_end - frame_start
            first_frame = dataset[frame_start]
            
            print(f"\n📊 Episode {selected_episode} Information:")
            print(f"  Task: {first_frame['task']}")
            print(f"  Frames: {frame_count}")
            print(f"  Duration: ~{frame_count / dataset.meta.fps:.1f} seconds")
            
            # Ask if user wants to visualize
            visualize = input("\nVisualize this episode? (y/n): ").strip().lower()
            
            if visualize in ['y', 'yes']:
                print(f"\nLaunching visualization for episode {selected_episode}...")
                print(f"Task: {first_frame['task']}")
                print(f"Task Index: {first_frame['task_index'].item()}")
                try:
                    visualize_ur5_dataset(
                        dataset=dataset,
                        episode_index=selected_episode,
                        batch_size=32,
                        num_workers=4
                    )
                except KeyboardInterrupt:
                    print("\nVisualization interrupted.")
                except Exception as e:
                    print(f"Error during visualization: {e}")
            
            # Ask if user wants to continue
            continue_exploring = input("\nExplore another episode? (y/n): ").strip().lower()
            if continue_exploring not in ['y', 'yes']:
                break
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("Thanks for exploring the UR5 dataset! 🤖")
    return 0


if __name__ == "__main__":
    # Setup logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    exit(main())