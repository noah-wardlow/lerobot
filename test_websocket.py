#!/usr/bin/env python3
import asyncio
import websockets

async def test_connection():
    try:
        websocket = await websockets.connect('ws://127.0.0.1:3201')
        print("✓ Websocket connection successful!")
        await websocket.close()
    except Exception as e:
        print(f"✗ Websocket connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())