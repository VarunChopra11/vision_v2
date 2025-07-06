import asyncio
import base64
from google import genai
import re
from typing import Dict, Set, Optional
import logging
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Audio constants
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini + PyAudio setup
MODEL = "gemini-live-2.5-flash-preview"
CONFIG = {
    "response_modalities": ["AUDIO"],
    "output_audio_transcription": {},
    "input_audio_transcription": {},
    "speech_config": {
        "language_code": "en-IN",
        "voice_config": {
            "prebuilt_voice_config": {
                "voice_name": "Aoede"  # Try: Charon, Kore, Fenrir, Aoede
            }
        }
    },
    "system_instruction": """(
                    "You are Vision, an intelligent real-time assistant designed to help visually impaired users "
                    "through voice and visual understanding.\n\n"
                    "You have access to a first-person video feed from the user's environment, allowing you to interpret "
                    "surroundings, objects, people, text, and obstacles in real time.\n\n"
                    "Your purpose is to provide clear, accurate, and helpful guidance to support the user in:\n"
                    "- Navigating safely\n"
                    "- Identifying objects and people\n"
                    "- Reading signs, text, or labels\n"
                    "- Describing surroundings\n"
                    "- Answering spoken queries\n"
                    "- Offering any other helpful contextual information\n\n"
                    "Always communicate in a calm, descriptive, and supportive manner. Your responses must prioritize "
                    "safety, clarity, and ease of understanding for someone who cannot see."
                )"""
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClientManager:
    def __init__(self):
        self.clients: Dict[str, 'AudioLoop'] = {}
        self.broadcaster_id: Optional[str] = None
        self.receivers: Set[str] = set()
        
    def add_client(self, client_id: str, client: 'AudioLoop'):
        """Add a new client to the manager"""
        self.clients[client_id] = client
        logger.info(f"Client {client_id} added. Total clients: {len(self.clients)}")
    
    def remove_client(self, client_id: str):
        """Remove a client from the manager"""
        if client_id in self.clients:
            del self.clients[client_id]
            
        if self.broadcaster_id == client_id:
            self.broadcaster_id = None
            logger.info(f"Broadcaster {client_id} disconnected")
            
        self.receivers.discard(client_id)
        logger.info(f"Client {client_id} removed. Total clients: {len(self.clients)}")
    
    def set_broadcaster(self, client_id: str) -> bool:
        """Set a client as the broadcaster"""
        if client_id not in self.clients:
            return False
            
        # Remove from receivers if was a receiver
        self.receivers.discard(client_id)
        
        # If there's already a broadcaster, demote them to receiver
        if self.broadcaster_id and self.broadcaster_id in self.clients:
            self.receivers.add(self.broadcaster_id)
            
        self.broadcaster_id = client_id
        logger.info(f"Client {client_id} is now the broadcaster")
        return True
    
    def add_receiver(self, client_id: str) -> bool:
        """Add a client as a receiver"""
        if client_id not in self.clients or client_id == self.broadcaster_id:
            return False
            
        self.receivers.add(client_id)
        logger.info(f"Client {client_id} is now a receiver")
        return True
    
    async def broadcast_to_receivers(self, message: dict, exclude_client: str = None):
        """Broadcast a message to all receivers"""
        if not self.receivers:
            return
            
        disconnected_clients = []
        
        for receiver_id in self.receivers:
            if receiver_id == exclude_client:
                continue
                
            if receiver_id in self.clients:
                try:
                    await self.clients[receiver_id].websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to receiver {receiver_id}: {e}")
                    disconnected_clients.append(receiver_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.remove_client(client_id)
    
    async def broadcast_to_all(self, message: dict, exclude_client: str = None):
        """Broadcast a message to all clients"""
        disconnected_clients = []
        
        for client_id, client in self.clients.items():
            if client_id == exclude_client:
                continue
                
            try:
                await client.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.remove_client(client_id)

# Global client manager instance
client_manager = ClientManager()

class AudioLoop:
    def __init__(self, client_id: str, websocket=None):
        self.client_id = client_id
        self.websocket = websocket
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.session_context = None
        self.client = None
        self.is_broadcaster = False
        self.is_receiver = False
        self.broadcaster_tasks = []  # Track broadcaster-specific tasks
        
        # Add this client to the manager
        client_manager.add_client(client_id, self)
    
    async def create_gemini_session(self):
        """Create a new Gemini session"""
        try:
            self.client = genai.Client(api_key = GEMINI_API_KEY, http_options={"api_version": "v1beta"})
            self.session_context = self.client.aio.live.connect(model=MODEL, config=CONFIG)
            self.session = await self.session_context.__aenter__()
            self.audio_in_queue = asyncio.Queue()
            self.out_queue = asyncio.Queue(maxsize=2)  # Smaller queue for faster processing
            
            # Start broadcaster tasks
            self.broadcaster_tasks = [
                asyncio.create_task(self.send_realtime()),
                asyncio.create_task(self.receive_audio())
            ]
            
            logger.info(f"Gemini session created for broadcaster {self.client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Gemini session: {e}")
            return False
    
    async def close_gemini_session(self):
        """Close the Gemini session and clean up"""
        # Cancel broadcaster tasks
        for task in self.broadcaster_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.broadcaster_tasks:
            await asyncio.gather(*self.broadcaster_tasks, return_exceptions=True)
        
        self.broadcaster_tasks = []
        
        # Close session
        if self.session and hasattr(self, 'session_context'):
            try:
                await self.session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            finally:
                self.session = None
                self.session_context = None
                
        logger.info(f"Gemini session closed for {self.client_id}")

    async def send_realtime(self):
        """Send messages to Gemini (only for broadcaster)"""
        if not self.is_broadcaster or not self.session:
            return
            
        try:
            while True:
                msg = await self.out_queue.get()
                await self.session.send(input=msg)
        except asyncio.CancelledError:
            logger.info(f"Send task cancelled for {self.client_id}")
        except Exception as e:
            logger.error(f"Error sending to Gemini: {e}")

    async def receive_audio(self):
        """Receive responses from Gemini (only for broadcaster)"""
        if not self.is_broadcaster or not self.session:
            return
            
        try:
            ai_buffer = ""
            while True:
                turn = self.session.receive()
                async for response in turn:
                    if response.server_content and response.server_content.output_transcription:
                        ai_text = response.server_content.output_transcription.text
                        ai_buffer += ai_text
                        
                        # Process complete sentences
                        matches = re.findall(r".*?[.?!](?:\s|$)", ai_buffer)
                        for match in matches:
                            ai_message = {
                                "type": "ai",
                                "data": match.strip()
                            }
                            # Send to broadcaster
                            await self.websocket.send_json(ai_message)
                            # Broadcast to all receivers
                            await client_manager.broadcast_to_receivers(ai_message, self.client_id)
                            
                        ai_buffer = ai_buffer[len("".join(matches)):]
                    
                    if response.server_content.input_transcription:
                        user_message = {
                            "type": "user",
                            "data": response.server_content.input_transcription.text
                        }
                        # Send to broadcaster
                        await self.websocket.send_json(user_message)
                        # Broadcast to all receivers
                        await client_manager.broadcast_to_receivers(user_message, self.client_id)
                        
                    elif response.data:
                        # Send Gemini's audio response
                        audio_bytes = response.data
                        encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
                        audio_message = {
                            "type": "audio_from_gemini",
                            "data": encoded_audio,
                            "sample_rate": RECEIVE_SAMPLE_RATE,
                            "format": "int16"
                        }
                        # Send to broadcaster
                        await self.websocket.send_json(audio_message)
                        # Broadcast to all receivers
                        await client_manager.broadcast_to_receivers(audio_message, self.client_id)

                # Drain old audio queue
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
                    
        except asyncio.CancelledError:
            logger.info(f"Receive task cancelled for {self.client_id}")
        except Exception as e:
            logger.error(f"Error receiving from Gemini: {e}")

    async def handle_client_stream(self):
        """Handle incoming messages from WebSocket"""
        while True:
            try:
                message = await self.websocket.receive_json()
                msg_type = message.get("type")
                
                if msg_type == "set_role":
                    # Client wants to set their role
                    role = message.get("role")
                    if role == "broadcaster":
                        # First, close any existing session if switching roles
                        if self.session:
                            await self.close_gemini_session()
                        
                        success = client_manager.set_broadcaster(self.client_id)
                        if success:
                            self.is_broadcaster = True
                            self.is_receiver = False
                            
                            # Create Gemini session
                            session_created = await self.create_gemini_session()
                            if session_created:
                                await self.websocket.send_json({
                                    "type": "role_confirmed",
                                    "role": "broadcaster"
                                })
                                # Notify all other clients about new broadcaster
                                await client_manager.broadcast_to_all({
                                    "type": "broadcaster_changed",
                                    "broadcaster_id": self.client_id
                                }, self.client_id)
                            else:
                                await self.websocket.send_json({
                                    "type": "role_error",
                                    "message": "Failed to create Gemini session"
                                })
                        else:
                            await self.websocket.send_json({
                                "type": "role_error",
                                "message": "Failed to set as broadcaster"
                            })
                    elif role == "receiver":
                        # Close session if switching from broadcaster
                        if self.session:
                            await self.close_gemini_session()
                        
                        success = client_manager.add_receiver(self.client_id)
                        if success:
                            self.is_receiver = True
                            self.is_broadcaster = False
                            await self.websocket.send_json({
                                "type": "role_confirmed",
                                "role": "receiver"
                            })
                        else:
                            await self.websocket.send_json({
                                "type": "role_error",
                                "message": "Failed to set as receiver"
                            })
                
                elif msg_type == "frame":
                    if self.is_broadcaster and self.session:
                        # Frame throttling - skip frames if queue is full
                        if self.out_queue.qsize() >= 3:  # Skip if queue has 3+ items
                            logger.debug("Skipping frame - queue full")
                            continue
                            
                        # Send frame to Gemini
                        image_bytes = base64.b64decode(message["data"])
                        
                        # Skip if frame is too small (likely corrupted)
                        if len(image_bytes) < 1000:
                            logger.debug("Skipping small frame")
                            continue
                            
                        # Clear old frames from queue before adding new one
                        while not self.out_queue.empty():
                            try:
                                old_item = self.out_queue.get_nowait()
                                if old_item.get("mime_type") == "image/jpeg":
                                    logger.debug("Removed old frame from queue")
                            except Exception as e:
                                print(f"Error clearing queue: {e}")
                                break
                        
                        # Send frame to Gemini
                        await self.out_queue.put({
                            "data": image_bytes,
                            "mime_type": "image/jpeg"
                        })
                        
                        logger.debug(f"Frame sent to Gemini (size: {len(image_bytes)} bytes)")
                elif msg_type == "frame-to-show":
                    # Broadcast frame to all receivers
                        frame_message = {
                            "type": "frame-to-show-frontend",
                            "data": message["data"]  # Use original base64 data
                        }
                        await client_manager.broadcast_to_receivers(frame_message, self.client_id)
                elif msg_type == "audio":
                    if self.is_broadcaster and self.session:
                        # Send audio to Gemini
                        audio_data = base64.b64decode(message["data"])
                        await self.out_queue.put({
                            "data": audio_data,
                            "mime_type": "audio/pcm"
                        })
                
                elif msg_type == "get_status":
                    # Send current status to client
                    await self.websocket.send_json({
                        "type": "status",
                        "client_id": self.client_id,
                        "is_broadcaster": self.is_broadcaster,
                        "is_receiver": self.is_receiver,
                        "broadcaster_id": client_manager.broadcaster_id,
                        "total_clients": len(client_manager.clients),
                        "receivers": list(client_manager.receivers),
                        "session_active": self.session is not None
                    })
                
                elif msg_type == "disconnect":
                    break
                    
            except Exception as e:
                logger.error(f"[Client Stream Error] {e}")
                break

    async def run_with_websocket(self):
        """Main entry point for running the audio loop with WebSocket"""
        try:
            # Just handle client stream - session will be created when role is set
            await self.handle_client_stream()
                
        except Exception as e:
            logger.error(f"Error in run_with_websocket: {e}")
            try:
                await self.websocket.send_json({
                    "type": "error", 
                    "data": f"Connection error: {str(e)}"
                })
            except Exception as e:
                print("WebSocket closed unexpectedly", e)
                pass  # WebSocket might be closed
        finally:
            # Clean up when client disconnects
            if self.session:
                await self.close_gemini_session()
            client_manager.remove_client(self.client_id)

# Factory function to create AudioLoop instances
async def create_audio_loop(client_id: str, websocket):
    """Factory function to create and run an AudioLoop instance"""
    audio_loop = AudioLoop(client_id, websocket)
    await audio_loop.run_with_websocket()
    return audio_loop