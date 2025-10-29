from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from gradio_client import Client
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen-backend")

app = FastAPI(
    title="Qwen Chat Backend API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gradio Client
HF_SPACE_NAME = "Redhanuman/qwen-chat-api"
HF_TOKEN = os.getenv("HF_TOKEN", "")

try:
    if HF_TOKEN:
        client = Client(HF_SPACE_NAME, hf_token=HF_TOKEN)
    else:
        client = Client(HF_SPACE_NAME)
    
    # View API info at startup
    logger.info("="*60)
    logger.info("GRADIO API DOCUMENTATION:")
    logger.info("="*60)
    client.view_api()
    logger.info(f"✅ Connected to {HF_SPACE_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to connect: {e}")
    client = None

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[List[str]]] = []

class ChatResponse(BaseModel):
    response: str
    success: bool
    message: Optional[str] = None

@app.get("/")
async def root():
    return {
        "status": "online",
        "space": HF_SPACE_NAME,
        "message": "Use /api/chat to send messages"
    }

@app.get("/api/health")
async def health_check():
    if client is None:
        return {"status": "unhealthy", "error": "Client not initialized"}
    
    try:
        # Simple test call
        result = client.predict("ping", api_name="/chat")
        return {
            "status": "healthy",
            "space": HF_SPACE_NAME,
            "connected": True
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="Client unavailable")
    
    try:
        logger.info(f"Message: {request.message[:50]}...")
        
        # ✅ Pass parameters as positional arguments, not keyword args
        result = client.predict(
            request.message,      # arg_0: message
            request.history,      # arg_1: history
            api_name="/chat"
        )
        
        logger.info(f"Response: {str(result)[:50]}...")
        
        return ChatResponse(
            response=result,
            success=True,
            message="Success"
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
