import os
import json
import asyncio
import websockets

class UnifiedInferenceServer:
    # ... your __init__ and other methods ...
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection."""
        print(f"✓ Client connected from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data['type'] == 'predict_finger':
                        # Finger reading prediction
                        if 'sensor' not in data:
                            raise ValueError("No sensor data")
                        
                        sensor_data = data['sensor']
                        
                        if 'finger' not in sensor_data or 'imu' not in sensor_data:
                            raise ValueError("Missing finger or IMU data")
                        
                        if len(sensor_data['finger']) == 0 or len(sensor_data['imu']) == 0:
                            raise ValueError("Empty sensor data")
                        
                        print(f"📊 Finger prediction: {len(sensor_data['finger'])} samples")
                        
                        predictions = self.predict_finger(sensor_data, top_k=5)
                        
                        print(f"✓ Top prediction: {predictions[0]['text'][:50]}...")
                        
                        response = {
                            'type': 'prediction',
                            'mode': 'finger',
                            'predictions': predictions
                        }
                        await websocket.send(json.dumps(response))
                    
                    elif data['type'] == 'recognize_braille':
                        # Braille recognition
                        if 'image' not in data:
                            raise ValueError("No image data")
                        
                        # Get optional finger position
                        finger_x = data.get('finger_x', None)
                        finger_y = data.get('finger_y', None)
                        
                        print(f"🔤 Braille recognition request (finger: {finger_x is not None})")
                        
                        result = self.recognize_braille(data['image'], finger_x, finger_y)
                        
                        print(f"✓ Detected {result['dotCount']} dots, {result.get('cellCount', 0)} cells")
                        if result.get('text'):
                            print(f"   Recognized: '{result['text']}'")
                        
                        response = {
                            'type': 'braille_result',
                            'mode': 'braille',
                            'result': result
                        }
                        await websocket.send(json.dumps(response))
                    
                    elif data['type'] == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                
                except ValueError as e:
                    print(f"⚠️  Validation error: {e}")
                    error_response = {
                        'type': 'error',
                        'message': str(e)
                    }
                    await websocket.send(json.dumps(error_response))
                
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    error_response = {
                        'type': 'error',
                        'message': str(e)
                    }
                    await websocket.send(json.dumps(error_response))
        
        except websockets.exceptions.ConnectionClosed:
            print(f"✗ Client disconnected")
        except Exception as e:
            print(f"❌ Connection error: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self, host: str = None, port: int = None):
        """Start WebSocket server."""
        # Get host and port from environment or use defaults
        if host is None:
            host = os.getenv('HOST', '0.0.0.0')
        if port is None:
            port = int(os.getenv('PORT', '8765'))
        
        print(f"\n🚀 Starting Unified Inference Server")
        print(f"🌐 WebSocket server on ws://{host}:{port}")
        print(f"Supports: Finger Reading + Braille Recognition")
        print(f"Press Ctrl+C to stop\n")
        
        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()  # Run forever

# At the bottom of your file:
if __name__ == "__main__":
    server = UnifiedInferenceServer(
        model_path='models/best_model.pt',
        index_path='models/inference_index.pkl',
        device='cpu'
    )
    
    # Start server - will use Railway's PORT automatically
    asyncio.run(server.start())