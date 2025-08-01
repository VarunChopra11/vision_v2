from fastapi import APIRouter, WebSocket
from app.services.geminiLive import create_audio_loop, logger
import uuid
from fastapi.websockets import WebSocketDisconnect

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    
    try:
        await create_audio_loop(client_id, websocket)
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error with client {client_id}: {e}")

@router.get("/wakeup")
async def wakeup():
    """
    Simple endpoint to check if the service is running.
    Returns a simple message to confirm the API is awake.
    """
    return {"status": "awake", "message": "Service is up and running"}