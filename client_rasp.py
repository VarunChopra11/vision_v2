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
CHUNK = 1024
FRAME_RATE = 15  # Reduced FPS target for Pi
RESOLUTION = (640, 480)  # Maintain resolution but use efficient encoding

# Camera settings for brightness/quality
BRIGHTNESS = 70  # Increased brightness (0-100)
CONTRAST = 70    # Increased contrast (0-100)
QUALITY = 85     # JPEG quality (1-100)

class PiCameraWrapper:
    """Wrapper for efficient PiCamera capture on Raspberry Pi"""
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = FRAME_RATE
        self.camera.brightness = BRIGHTNESS
        self.camera.contrast = CONTRAST
        self.raw_capture = PiRGBArray(self.camera, size=RESOLUTION)
        self.stream = self.camera.capture_continuous(
            self.raw_capture, 
            format="bgr", 
            use_video_port=True
        )
        
    def read(self):
        frame = next(self.stream).array
        self.raw_capture.truncate(0)
        return True, frame
    
    def release(self):
        self.stream.close()
        self.raw_capture.close()
        self.camera.close()

class OpenCVCamera:
    """Fallback camera for non-Raspberry Pi systems"""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS/100)
        self.cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST/100)
        
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame
    
    def release(self):
        self.cap.release()


async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def encode_frame(frame):
    """Efficient JPEG encoding with brightness adjustment"""
    # Adjust brightness/contrast
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    
    # Encode directly to JPEG
    _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
    return base64.b64encode(jpeg).decode('utf-8')


async def send_audio_and_video(uri):
    audio = pyaudio.PyAudio()
    stream_send = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    stream_play = None
    is_playing_audio = asyncio.Event()

    # Initialize camera
    if USE_PICAMERA:
        print("Using PiCamera for optimized performance")
        camera = PiCameraWrapper()
    else:
        print("Using OpenCV camera")
        camera = OpenCVCamera()

    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
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

            print("Starting streams...")
            last_frame_time = time.time()

            async def send_audio():
                """Send audio chunks when not playing back AI audio"""
                try:
                    while True:
                        if is_playing_audio.is_set():
                            await asyncio.sleep(0.05)
                            continue
                        data = await run_in_thread(
                            stream_send.read, CHUNK, exception_on_overflow=False
                        )
                        encoded = base64.b64encode(data).decode("utf-8")
                        await ws.send(json.dumps({"type": "audio", "data": encoded}))
                except Exception as e:
                    print(f"Audio send error: {e}")

            async def send_video():
                """Optimized video streaming with FPS control"""
                try:
                    while True:
                        start_time = time.time()
                        success, frame = await run_in_thread(camera.read)
                        
                        if not success:
                            await asyncio.sleep(0.01)
                            continue
                            
                        encoded = await run_in_thread(encode_frame, frame)
                        
                        # Send both frame types efficiently
                        await ws.send(json.dumps({"type": "frame", "data": encoded}))
                        await ws.send(json.dumps({"type": "frame-to-show", "data": encoded}))
                        
                        # Dynamic sleep for FPS control
                        elapsed = time.time() - start_time
                        sleep_time = max(0, (1.0 / FRAME_RATE) - elapsed)
                        await asyncio.sleep(sleep_time)
                        
                except Exception as e:
                    print(f"Video send error: {e}")

            async def receive_messages():
                """Handle incoming messages from server"""
                nonlocal stream_play
                try:
                    while True:
                        message = await ws.recv()
                        msg_json = json.loads(message)
                        msg_type = msg_json.get("type")

                        if msg_type == "audio_from_gemini":
                            is_playing_audio.set()
                            audio_data = base64.b64decode(msg_json["data"])
                            sample_rate = msg_json.get("sample_rate", 24000)

                            if stream_play is None:
                                stream_play = audio.open(
                                    format=FORMAT,
                                    channels=CHANNELS,
                                    rate=sample_rate,
                                    output=True,
                                )

                            await run_in_thread(stream_play.write, audio_data)
                            is_playing_audio.clear()
                            print("🔊 Playing audio")

                        elif msg_type == "ai":
                            print(f"🤖 AI: {msg_json['data']}")
                        elif msg_type == "error":
                            print(f"❌ Error: {msg_json['data']}")
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket closed")

            # Run all tasks concurrently
            await asyncio.gather(
                send_audio(),
                send_video(),
                receive_messages(),
            )

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Cleaning up...")
        stream_send.stop_stream()
        stream_send.close()
        if stream_play:
            stream_play.stop_stream()
            stream_play.close()
        audio.terminate()
        camera.release()


if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        print("Exit")