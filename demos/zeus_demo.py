import asyncio
import websockets
import json

async def websocket_client():
    uri = "ws://192.168.4.1:8765"  # ESP32-CAM IP address when it creates an Access Point
    async with websockets.connect(uri) as websocket:

        message = json.dumps({"A": -1.5708, "B": 0.8, "C": 0.0, "D": 1})
        print(message)
        await websocket.send(message+">")
        print(f"\n> {message}\n")

        # Waiting for a response (e.g., "pong")
        response = await websocket.recv()
        print(f"\n< {response}\n")

        # Waiting for a response (e.g., "pong")
        response = await websocket.recv()
        print(f"\n< {response}\n")
        
        response = await websocket.recv()
        print(f"\n< {response}\n")

# Running the client
if __name__ == "__main__":
    asyncio.run(websocket_client())
    