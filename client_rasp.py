import asyncio
import websockets
import pyaudio
import base64
import os
import json
from dotenv import load_dotenv
import time
import logging

# Conditional import for PiCamera
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    USE_PICAMERA = True
except (ImportError, OSError):
    USE_PICAMERA = False
    import cv2

load_dotenv(override=True)

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WEBSOCKET_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8000/ws")
CHUNK = 1024
FRAME_RATE = 15  # Optimized for Pi 3B
RESOLUTION = (640, 480)
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 80
MAX_QUEUE_SIZE = 2  # Backpressure control
AUDIO_BUFFER_SIZE = 1024 * 4  # Smaller buffer for lower latency

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
    def read(self):
        try:
            frame = next(self.stream)
            jpeg_data = frame.array
            self.raw_capture.truncate(0)
            return True, jpeg_data
        except StopIteration:
            return False, None
        except Exception as e:
            logger.error(f"Camera read error: {e}")
            return False, None
    
    def release(self):
        try:
            self.stream.close()
            self.raw_capture.close()
            self.camera.close()
        except Exception as e:
            logger.error(f"Camera release error: {e}")

class OpenCVCamera:
    """Fallback camera for non-Raspberry Pi systems"""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS/100)
        self.cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST/100)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.25)    # Increase exposure for brightness
        
    def read(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                return False, None
            
            # Adjust brightness/contrast
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=25)
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
            return True, jpeg.tobytes()
        except Exception as e:
            logger.error(f"OpenCV camera error: {e}")
            return False, None
    
    def release(self):
        try:
            self.cap.release()
        except Exception as e:
            logger.error(f"OpenCV release error: {e}")


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
    camera = None
    try:
        if USE_PICAMERA:
            logger.info("Using PiCamera for optimized performance")
            camera = PiCameraWrapper()
        else:
            logger.info("Using OpenCV camera")
            camera = OpenCVCamera()
    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        return

    logger.info(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
            logger.info("Connected! Setting role as broadcaster...")
            await ws.send(json.dumps({"type": "set_role", "role": "broadcaster"}))

            # Wait for role confirmation
            role_confirmed = False
            while not role_confirmed:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg_json = json.loads(message)
                    if msg_json.get("type") == "role_confirmed":
                        logger.info(f"Role confirmed: {msg_json.get('role')}")
                        role_confirmed = True
                    elif msg_json.get("type") == "role_error":
                        logger.error(f"Role error: {msg_json.get('message')}")
                        return
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for role confirmation")
                    return

            logger.info("Starting streams...")

            async def capture_video():
                """Dedicated video capture with timing control"""
                try:
                    target_delay = 1.0 / FRAME_RATE
                    while True:
                        start_capture = time.monotonic()
                        
                        success, jpeg_data = await run_in_thread(camera.read)
                        if not success:
                            await asyncio.sleep(0.01)
                            continue
                            
                        try:
                            video_queue.put_nowait(jpeg_data)
                        except asyncio.QueueFull:
                            # Drop frame if queue full to prevent backlog
                            pass
                            
                        # Calculate time to maintain frame rate
                        elapsed = time.monotonic() - start_capture
                        sleep_time = max(0, target_delay - elapsed)
                        await asyncio.sleep(sleep_time)
                        
                except Exception as e:
                    logger.error(f"Video capture error: {e}")

            async def send_audio():
                """Audio streaming with precise timing"""
                try:
                    chunk_duration = CHUNK / RATE  # Time per chunk in seconds
                    last_send_time = time.monotonic()
                    
                    while True:
                        # Skip audio if playing back Gemini audio
                        if is_playing_audio.is_set():
                            await asyncio.sleep(0.01)
                            continue
                            
                        # Read audio data
                        data = await run_in_thread(
                            stream_send.read, CHUNK, exception_on_overflow=False
                        )
                        encoded = base64.b64encode(data).decode("utf-8")
                        
                        # Send with precise timing
                        await ws.send(json.dumps({"type": "audio", "data": encoded}))
                        
                        # Maintain exact timing
                        current_time = time.monotonic()
                        elapsed = current_time - last_send_time
                        sleep_time = max(0, chunk_duration - elapsed)
                        await asyncio.sleep(sleep_time)
                        last_send_time = time.monotonic() + max(0, elapsed - chunk_duration)
                        
                except Exception as e:
                    logger.error(f"Audio send error: {e}")

            async def send_video():
                """Video sending with queue-based backpressure and FPS monitoring"""
                start_time = time.monotonic()
                frame_count = 0
                
                try:
                    while True:
                        jpeg_data = await video_queue.get()
                        encoded = base64.b64encode(jpeg_data).decode('utf-8')
                        
                        # Send both frame types efficiently
                        await asyncio.gather(
                            ws.send(json.dumps({"type": "frame", "data": encoded})),
                            ws.send(json.dumps({"type": "frame-to-show", "data": encoded}))
                        )
                        
                        # Report FPS periodically
                        frame_count += 1
                        elapsed = time.monotonic() - start_time
                        if elapsed > 5.0:
                            fps = frame_count / elapsed
                            logger.info(f"📹 Streaming at {fps:.1f} FPS")
                            frame_count = 0
                            start_time = time.monotonic()
                            
                except Exception as e:
                    logger.error(f"Video send error: {e}")

            async def receive_messages():
                """Handle incoming messages from server with low-latency audio"""
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
                                    frames_per_buffer=AUDIO_BUFFER_SIZE  # Smaller buffer for lower latency
                                )

                            # Write in thread to avoid blocking
                            await run_in_thread(stream_play.write, audio_data)
                            is_playing_audio.clear()
                            logger.info("🔊 Playing audio")

                        elif msg_type == "ai":
                            logger.info(f"🤖 AI: {msg_json['data']}")
                        elif msg_type == "error":
                            logger.error(f"❌ Error: {msg_json['data']}")
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket closed")
                except Exception as e:
                    logger.error(f"Message receive error: {e}")

            # Run all tasks concurrently
            await asyncio.gather(
                capture_video(),
                send_audio(),
                send_video(),
                receive_messages(),
            )

    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        logger.info("Cleaning up resources...")
        try:
            stream_send.stop_stream()
            stream_send.close()
        except:
            pass
            
        if stream_play:
            try:
                stream_play.stop_stream()
                stream_play.close()
            except:
                pass
                
        audio.terminate()
        
        if camera:
            camera.release()
            
        logger.info("All resources released")


if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        logger.info("Exit")