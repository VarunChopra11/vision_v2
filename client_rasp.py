import asyncio
import websockets
import pyaudio
import base64
import cv2
import os
import json
from dotenv import load_dotenv
import time
import io
import numpy as np
from collections import deque
import threading

# Conditional import for PiCamera
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    USE_PICAMERA = True
except (ImportError, OSError):
    USE_PICAMERA = False

load_dotenv(override=True)

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WEBSOCKET_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8000/ws")
CHUNK = 512  # Reduced chunk size for better responsiveness
FRAME_RATE = 15
RESOLUTION = (640, 480)

# Camera settings
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 85

# Audio buffer settings
AUDIO_BUFFER_SIZE = 10  # Maximum audio chunks to buffer
PLAYBACK_BUFFER_SIZE = 3  # Chunks to buffer before starting playback

class AudioManager:
    """Manages audio input/output with proper synchronization"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_playing = asyncio.Event()
        self.audio_queue = deque(maxlen=AUDIO_BUFFER_SIZE)
        self.playback_queue = deque()
        self.lock = threading.Lock()
        
    def initialize_streams(self):
        """Initialize audio streams with optimal settings"""
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._input_callback
        )
        
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=24000,  # Common AI audio sample rate
            output=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._output_callback
        )
        
    def _input_callback(self, in_data, frame_count, time_info, status):
        """Non-blocking audio input callback"""
        if not self.is_playing.is_set():
            with self.lock:
                if len(self.audio_queue) < AUDIO_BUFFER_SIZE:
                    self.audio_queue.append(in_data)
        return (None, pyaudio.paContinue)
    
    def _output_callback(self, in_data, frame_count, time_info, status):
        """Non-blocking audio output callback"""
        with self.lock:
            if self.playback_queue:
                data = self.playback_queue.popleft()
                return (data, pyaudio.paContinue)
        return (b'\x00' * frame_count * CHANNELS * 2, pyaudio.paContinue)
    
    def get_audio_data(self):
        """Get audio data for transmission"""
        with self.lock:
            if self.audio_queue:
                return self.audio_queue.popleft()
        return None
    
    def queue_playback_audio(self, audio_data):
        """Queue audio for playback"""
        with self.lock:
            self.playback_queue.append(audio_data)
    
    def start_playback(self):
        """Start AI audio playback"""
        self.is_playing.set()
        
    def stop_playback(self):
        """Stop AI audio playback"""
        self.is_playing.clear()
        
    def cleanup(self):
        """Clean up audio resources"""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()

class PiCameraWrapper:
    """Optimized PiCamera wrapper"""
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = FRAME_RATE
        self.camera.brightness = BRIGHTNESS
        self.camera.contrast = CONTRAST
        self.camera.sensor_mode = 7  # Fast mode for better performance
        self.raw_capture = PiRGBArray(self.camera, size=RESOLUTION)
        self.stream = self.camera.capture_continuous(
            self.raw_capture, 
            format="bgr", 
            use_video_port=True
        )
        
    def read(self):
        try:
            frame = next(self.stream).array
            self.raw_capture.truncate(0)
            return True, frame
        except StopIteration:
            return False, None
    
    def release(self):
        self.stream.close()
        self.raw_capture.close()
        self.camera.close()

class OpenCVCamera:
    """Optimized OpenCV camera wrapper"""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS/100)
        self.cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST/100)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
    def read(self):
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        self.cap.release()

class FrameEncoder:
    """Efficient frame encoding with caching"""
    def __init__(self):
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]
        
    def encode_frame(self, frame):
        """Optimized frame encoding"""
        # Enhance frame quality
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=15)
        
        # Encode to JPEG
        success, jpeg = cv2.imencode('.jpg', frame, self.encode_params)
        if not success:
            return None
            
        return base64.b64encode(jpeg).decode('utf-8')

async def send_audio_and_video(uri):
    audio_manager = AudioManager()
    frame_encoder = FrameEncoder()
    
    # Initialize camera
    if USE_PICAMERA:
        print("Using PiCamera for optimized performance")
        camera = PiCameraWrapper()
    else:
        print("Using OpenCV camera")
        camera = OpenCVCamera()

    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=30) as ws:
            print("Connected! Setting role as broadcaster...")
            await ws.send(json.dumps({"type": "set_role", "role": "broadcaster"}))

            # Wait for role confirmation
            role_confirmed = False
            while not role_confirmed:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg_json = json.loads(message)
                    if msg_json.get("type") == "role_confirmed":
                        print(f"Role confirmed: {msg_json.get('role')}")
                        role_confirmed = True
                    elif msg_json.get("type") == "role_error":
                        print(f"Role error: {msg_json.get('message')}")
                        return
                except asyncio.TimeoutError:
                    print("Timeout waiting for role confirmation")
                    return

            # Initialize audio streams
            audio_manager.initialize_streams()
            print("Starting streams...")

            async def send_audio():
                """Optimized audio sending with proper buffering"""
                try:
                    while True:
                        audio_data = audio_manager.get_audio_data()
                        if audio_data:
                            encoded = base64.b64encode(audio_data).decode("utf-8")
                            await ws.send(json.dumps({"type": "audio", "data": encoded}))
                        await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                except Exception as e:
                    print(f"Audio send error: {e}")

            async def send_video():
                """Optimized video streaming with frame skipping"""
                try:
                    frame_skip_counter = 0
                    target_interval = 1.0 / FRAME_RATE
                    
                    while True:
                        start_time = time.time()
                        
                        success, frame = camera.read()
                        if not success:
                            await asyncio.sleep(0.01)
                            continue
                            
                        # Skip frames if we're behind
                        frame_skip_counter += 1
                        if frame_skip_counter % 2 == 0:  # Send every other frame if needed
                            encoded = frame_encoder.encode_frame(frame)
                            if encoded:
                                # Send frame data
                                await ws.send(json.dumps({"type": "frame", "data": encoded}))
                                await ws.send(json.dumps({"type": "frame-to-show", "data": encoded}))
                        
                        # Adaptive timing
                        elapsed = time.time() - start_time
                        sleep_time = max(0.001, target_interval - elapsed)
                        await asyncio.sleep(sleep_time)
                        
                except Exception as e:
                    print(f"Video send error: {e}")

            async def receive_messages():
                """Handle incoming messages with reduced latency"""
                try:
                    while True:
                        message = await ws.recv()
                        msg_json = json.loads(message)
                        msg_type = msg_json.get("type")

                        if msg_type == "audio_from_gemini":
                            # Start playback immediately
                            audio_manager.start_playback()
                            
                            # Decode and queue audio
                            audio_data = base64.b64decode(msg_json["data"])
                            
                            # Split audio into smaller chunks for smoother playback
                            chunk_size = CHUNK * 2  # 2 bytes per sample
                            audio_chunks = [audio_data[i:i+chunk_size] 
                                          for i in range(0, len(audio_data), chunk_size)]
                            
                            # Queue all chunks
                            for chunk in audio_chunks:
                                if len(chunk) == chunk_size:  # Only queue full chunks
                                    audio_manager.queue_playback_audio(chunk)
                            
                            print("🔊 Playing audio")
                            
                            # Stop playback after a short delay
                            await asyncio.sleep(0.1)
                            audio_manager.stop_playback()

                        elif msg_type == "ai":
                            print(f"🤖 AI: {msg_json['data']}")
                        elif msg_type == "error":
                            print(f"❌ Error: {msg_json['data']}")
                            
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket closed")
                except Exception as e:
                    print(f"Receive error: {e}")

            # Run all tasks concurrently
            await asyncio.gather(
                send_audio(),
                send_video(),
                receive_messages(),
                return_exceptions=True
            )

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Cleaning up...")
        audio_manager.cleanup()
        camera.release()

if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        print("Exit")