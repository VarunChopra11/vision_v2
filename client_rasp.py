import asyncio
import websockets
import pyaudio
import base64
import os
import json
from dotenv import load_dotenv
import time
import cv2
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
FRAME_RATE = 12  # Optimized for Pi 3B
RESOLUTION = (640, 480)
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 80
MAX_QUEUE_SIZE = 2  # Backpressure control

class PiCameraWrapper:
    """Optimized PiCamera capture with hardware encoding"""
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = FRAME_RATE
        self.camera.brightness = BRIGHTNESS
        self.camera.contrast = CONTRAST
        self.raw_capture = PiRGBArray(self.camera, size=RESOLUTION)
        self.stream = self.camera.capture_continuous(
            self.raw_capture, 
            format="jpeg",  # Use JPEG for hardware acceleration
            use_video_port=True,
            quality=QUALITY
        )
        self.last_frame_time = time.time()
        
    def read(self):
        frame = next(self.stream)
        jpeg_data = frame.array
        self.raw_capture.truncate(0)
        return True, jpeg_data
    
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
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.25)    # Increase exposure for brightness
        
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        
        # Adjust brightness/contrast
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=25)
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
        return True, jpeg.tobytes()
    
    def release(self):
        self.cap.release()


async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


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
    video_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

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
            frame_count = 0
            start_time = time.time()

            async def capture_video():
                """Dedicated video capture with timing control"""
                nonlocal last_frame_time, frame_count
                try:
                    while True:
                        current_time = time.time()
                        elapsed = current_time - last_frame_time
                        target_interval = 1.0 / FRAME_RATE
                        
                        if elapsed < target_interval:
                            await asyncio.sleep(target_interval - elapsed)
                            
                        success, jpeg_data = await run_in_thread(camera.read)
                        if not success:
                            await asyncio.sleep(0.01)
                            continue
                            
                        try:
                            video_queue.put_nowait(jpeg_data)
                        except asyncio.QueueFull:
                            # Drop frame if queue full to prevent backlog
                            pass
                            
                        last_frame_time = time.time()
                        frame_count += 1
                        
                except Exception as e:
                    print(f"Video capture error: {e}")

            async def send_audio():
                """Audio streaming with backpressure"""
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
                        
                        # Maintain audio timing
                        await asyncio.sleep(CHUNK / RATE)
                        
                except Exception as e:
                    print(f"Audio send error: {e}")

            async def send_video():
                """Video sending with queue-based backpressure"""
                try:
                    while True:
                        jpeg_data = await video_queue.get()
                        encoded = base64.b64encode(jpeg_data).decode('utf-8')
                        
                        # Send both frame types in one go
                        await asyncio.gather(
                            ws.send(json.dumps({"type": "frame", "data": encoded})),
                            ws.send(json.dumps({"type": "frame-to-show", "data": encoded}))
                        )
                        
                        # Report FPS periodically
                        if time.time() - start_time > 5:
                            fps = frame_count / (time.time() - start_time)
                            print(f"📹 Streaming at {fps:.1f} FPS")
                            frame_count = 0
                            start_time = time.time()
                            
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

                            stream_play.write(audio_data)
                            is_playing_audio.clear()
                            # print("🔊 Playing audio")

                        elif msg_type == "ai":
                            print(f"🤖 AI: {msg_json['data']}")
                        elif msg_type == "error":
                            print(f"❌ Error: {msg_json['data']}")
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket closed")

            # Run all tasks concurrently
            await asyncio.gather(
                capture_video(),
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
        print("Resources released")


if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        print("Exit")