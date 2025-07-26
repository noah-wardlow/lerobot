#!/usr/bin/env python3

"""
Lab simulation camera viewer for /scene_camera/color and /wrist_camera/color.
Connects to rosbridge and streams both cameras to Rerun visualization.
"""

import asyncio
import json
import logging
import numpy as np
import rerun as rr
import websockets
from typing import Dict, Any
import base64


class LabCameraStreamer:
    def __init__(self, rosbridge_host: str = "127.0.0.1", rosbridge_port: int = 3201):
        self.rosbridge_url = f"ws://{rosbridge_host}:{rosbridge_port}"
        self.websocket = None
        self.subscriptions = {}
        
    async def connect(self):
        """Connect to rosbridge websocket."""
        try:
            self.websocket = await websockets.connect(
                self.rosbridge_url,
                max_size=10*1024*1024  # 10MB limit for large images
            )
            logging.info(f"Connected to rosbridge at {self.rosbridge_url}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to rosbridge: {e}")
            return False
    
    async def subscribe_to_camera(self, topic_name: str, camera_name: str):
        """Subscribe to a camera image topic."""
        subscription_msg = {
            "op": "subscribe",
            "topic": topic_name,
            "type": "sensor_msgs/Image"
        }
        
        await self.websocket.send(json.dumps(subscription_msg))
        self.subscriptions[topic_name] = camera_name
        logging.info(f"Subscribed to camera topic: {topic_name} -> {camera_name}")
    
    def process_image_message(self, msg: Dict[str, Any]) -> np.ndarray:
        """Convert ROS2 Image message to numpy array."""
        try:
            # Extract image data
            width = msg["msg"]["width"]
            height = msg["msg"]["height"]
            encoding = msg["msg"]["encoding"]
            data = msg["msg"]["data"]
            
            # Decode base64 data
            image_bytes = base64.b64decode(data)
            
            # Convert to numpy array based on encoding
            if encoding == "rgb8":
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image_array = image_array.reshape((height, width, 3))
            elif encoding == "bgr8":
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image_array = image_array.reshape((height, width, 3))
                # BGR to RGB conversion
                image_array = image_array[:, :, ::-1]
            elif encoding == "mono8":
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image_array = image_array.reshape((height, width, 1))
                # Convert grayscale to RGB
                image_array = np.repeat(image_array, 3, axis=2)
            else:
                logging.warning(f"Unsupported encoding: {encoding}")
                return None
            
            return image_array
            
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return None
    
    async def stream_to_rerun(self):
        """Main streaming loop that sends camera data to Rerun."""
        frame_count = 0
        message_count = 0
        
        async for message in self.websocket:
            try:
                msg = json.loads(message)
                message_count += 1
                
                # Log first few messages for debugging
                if message_count <= 5:
                    logging.info(f"Message {message_count}: op={msg.get('op')}, topic={msg.get('topic')}")
                
                if msg.get("op") == "publish":
                    topic = msg.get("topic")
                    
                    if topic in self.subscriptions:
                        camera_name = self.subscriptions[topic]
                        
                        # Extract and set ROS timestamp if available
                        ros_time = None
                        if "stamp" in msg.get("msg", {}).get("header", {}):
                            stamp = msg["msg"]["header"]["stamp"]
                            ros_time = stamp["sec"] + stamp["nanosec"] * 1e-9
                            rr.set_time_seconds("ros_time", ros_time)
                        
                        # Set frame sequence
                        rr.set_time_sequence("frame", frame_count)
                        
                        # Process image
                        image_array = self.process_image_message(msg)
                        
                        # Log to Rerun with proper entity path
                        if image_array is not None:
                            entity_path = f"world/robot/{camera_name}"
                            rr.log(entity_path, rr.Image(image_array))
                            
                            frame_count += 1
                            if frame_count % 20 == 0:
                                time_str = f" at {ros_time:.3f}s" if ros_time else ""
                                logging.info(f"Streamed {frame_count} frames{time_str}")
                        else:
                            logging.warning(f"Failed to process image from {camera_name}")
                        
            except json.JSONDecodeError:
                logging.warning("Received invalid JSON message")
            except Exception as e:
                logging.error(f"Error processing message: {e}")


async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize Rerun recording
    rr.init("lab_simulation_cameras", spawn=True)
    
    # Log static scene setup
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/robot", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    logging.info("Rerun viewer should have opened automatically")
    logging.info("If not, visit: http://localhost:9876")
    
    # Create streamer
    streamer = LabCameraStreamer()
    
    # Connect to rosbridge
    if not await streamer.connect():
        return 1
    
    # Subscribe to both lab cameras
    await streamer.subscribe_to_camera("/scene_camera/color", "scene_camera")
    await streamer.subscribe_to_camera("/wrist_camera/color", "wrist_camera")
    
    logging.info("Streaming lab simulation cameras... Press Ctrl+C to stop")
    
    try:
        await streamer.stream_to_rerun()
    except KeyboardInterrupt:
        logging.info("Stopping camera stream...")
    except Exception as e:
        logging.error(f"Stream error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))