from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from gradio_client import Client
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen-backend")

app = FastAPI(title="Qwen Chat Backend API", version="3.2.0")

# Fix CORS to allow OPTIONS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly add OPTIONS
    allow_headers=["*"],
    expose_headers=["*"]
)

SPACE_NAME = "Redhanuman/qwen-chat-api"
client = None

try:
    client = Client(SPACE_NAME)
    logger.info(f"✅ Connected to {SPACE_NAME}")
    
    # View API at startup
    logger.info("="*60)
    logger.info("GRADIO API STRUCTURE:")
    logger.info("="*60)
    client.view_api()
    
except Exception as e:
    logger.error(f"❌ Connection failed: {e}")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    success: bool
    message: Optional[str] = None

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Qwen Chat Backend API",
        "space": SPACE_NAME,
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health():
    if not client:
        return {"status": "unhealthy", "error": "Client not initialized"}
    
    try:
        result = client.predict("test", api_name="/chat")
        return {"status": "healthy", "space": SPACE_NAME, "connected": True}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not client:
        raise HTTPException(503, "Service unavailable")
    
    try:
        logger.info(f"📩 Received: {req.message[:50]}...")
        
        # Use submit() + result() for proper queue handling
        job = client.submit(
            req.message,
            api_name="/chat"
        )
        
        # Wait for result (blocks until complete)
        result = job.result()
        
        logger.info(f"📤 Response: {str(result)[:100]}...")
        
        return ChatResponse(
            response=result,
            success=True,
            message="Success"
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
