import asyncio
import base64
import io
import os
import json
from PIL import Image
from dotenv import load_dotenv
import cv2
import pyaudio
import websockets

load_dotenv(override=True)

# Audio configuration
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
WEBSOCKET_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8000/ws")

# Video configuration
VIDEO_FRAME_RATE = 25  # FPS
VIDEO_RESOLUTION = (640, 480)


class AudioVideoClient:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream_send = None
        self.stream_play = None
        self.is_playing = asyncio.Event()
        self.camera = cv2.VideoCapture(0)

    async def connect(self):
        """Establish WebSocket connection and handle streams."""
        if not self._check_camera():
            raise RuntimeError("Camera not available")

        async with websockets.connect(WEBSOCKET_URI, max_size=None) as ws:
            await self._set_broadcaster_role(ws)
            await self._start_streams(ws)

    async def _set_broadcaster_role(self, ws):
        """Set client role as broadcaster and confirm with server."""
        await ws.send(json.dumps({"type": "set_role", "role": "broadcaster"}))
        
        response = await asyncio.wait_for(ws.recv(), timeout=5.0)
        msg = json.loads(response)
        
        if msg.get("type") == "role_error":
            raise RuntimeError(f"Role error: {msg.get('message')}")
        print(f"Role confirmed: {msg.get('role')}")

    async def _start_streams(self, ws):
        """Start audio/video streams and message handling."""
        self.stream_send = self._open_audio_stream(input=True)
        
        tasks = [
            self._send_audio(ws),
            self._send_video(ws),
            self._send_display_video(ws),
            self._receive_messages(ws)
        ]
        await asyncio.gather(*tasks)

    def _check_camera(self):
        """Verify camera is available."""
        if not self.camera.isOpened():
            print("Error: Could not open camera")
            return False
        return True

    def _open_audio_stream(self, input=False, output=False, rate=SAMPLE_RATE):
        """Open PyAudio stream with given configuration."""
        return self.audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=rate,
            input=input,
            output=output,
            frames_per_buffer=CHUNK_SIZE
        )

    async def _send_audio(self, ws):
        """Continuously send audio chunks to server."""
        try:
            while True:
                if self.is_playing.is_set():
                    await asyncio.sleep(0.05)
                    continue

                data = await asyncio.to_thread(
                    self.stream_send.read, CHUNK_SIZE, exception_on_overflow=False
                )
                await ws.send(json.dumps({
                    "type": "audio",
                    "data": base64.b64encode(data).decode('utf-8')
                }))
        except Exception as e:
            print(f"Audio send error: {e}")

    async def _send_video(self, ws):
        """Send video frames to server at reduced rate."""
        while True:
            frame = await self._get_camera_frame()
            if frame is not None:
                encoded = await self._encode_frame(frame)
                await ws.send(json.dumps({"type": "frame", "data": encoded}))
                await asyncio.sleep(1)  # Throttle to 1 FPS

    async def _send_display_video(self, ws):
        """Send video frames for display at full frame rate."""
        while True:
            frame = await self._get_camera_frame()
            if frame is not None:
                encoded = await self._encode_frame(frame)
                await ws.send(json.dumps({"type": "frame-to-show", "data": encoded}))
                await asyncio.sleep(1/VIDEO_FRAME_RATE)

    async def _get_camera_frame(self):
        """Capture frame from camera."""
        ret, frame = await asyncio.to_thread(self.camera.read)
        return frame if ret else None

    async def _encode_frame(self, frame):
        """Convert frame to base64-encoded JPEG."""
        def _process_frame(f):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail(VIDEO_RESOLUTION)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return await asyncio.to_thread(_process_frame, frame)

    async def _receive_messages(self, ws):
        """Handle incoming WebSocket messages."""
        try:
            while True:
                msg = json.loads(await ws.recv())
                handler = {
                    "audio_from_gemini": self._handle_audio,
                    "ai": lambda m: print(f"🤖 {m['data']}"),
                    "user": lambda m: print(f"👤 {m['data']}"),
                    "error": lambda m: print(f"❌ {m['data']}"),
                    "status": lambda m: print(f"📊 {m}"),
                    "broadcaster_changed": lambda m: print(f"📡 {m}")
                }.get(msg["type"], lambda _: None)
                
                handler(msg)
        except websockets.ConnectionClosed as e:
            print(f"Connection closed: {e}")

    def _handle_audio(self, msg):
        """Play received audio stream."""
        self.is_playing.set()
        
        if not self.stream_play:
            self.stream_play = self._open_audio_stream(
                output=True,
                rate=msg.get("sample_rate", 24000)
            )
        
        audio_data = base64.b64decode(msg["data"])
        asyncio.to_thread(self.stream_play.write, audio_data)
        self.is_playing.clear()

    async def close(self):
        """Clean up resources."""
        for stream in [self.stream_send, self.stream_play]:
            if stream:
                stream.stop_stream()
                stream.close()
        
        self.audio.terminate()
        self.camera.release()


async def main():
    client = AudioVideoClient()
    try:
        print(f"Connecting to {WEBSOCKET_URI}...")
        await client.connect()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program stopped by user")