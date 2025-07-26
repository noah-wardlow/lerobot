#!/usr/bin/env python3
import asyncio
import json
import websockets

async def list_topics():
    try:
        websocket = await websockets.connect('ws://127.0.0.1:3201')
        
        # Request topic list
        msg = {
            "op": "call_service",
            "service": "/rosapi/topics",
            "args": {}
        }
        
        await websocket.send(json.dumps(msg))
        
        response = await websocket.recv()
        result = json.loads(response)
        
        print("Available topics:")
        for topic in result.get("values", {}).get("topics", []):
            print(f"  {topic}")
        
        await websocket.close()
        
    except Exception as e:
        print(f"Error listing topics: {e}")

if __name__ == "__main__":
    asyncio.run(list_topics())