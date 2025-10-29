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
    description="Backend API for Qwen 2.5 Chat via Gradio Client",
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
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional for public spaces

try:
    if HF_TOKEN:
        client = Client(HF_SPACE_NAME, hf_token=HF_TOKEN)
    else:
        client = Client(HF_SPACE_NAME)
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
        "message": "Qwen Chat Backend API (Gradio Client)",
        "space": HF_SPACE_NAME
    }

@app.get("/api/health")
async def health_check():
    if client is None:
        return {
            "status": "unhealthy",
            "error": "Gradio client not initialized"
        }
    
    try:
        # Test with a simple ping
        result = client.predict("ping", [], api_name="/chat")
        return {
            "status": "healthy",
            "space": HF_SPACE_NAME,
            "connected": True
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Gradio client not available"
        )
    
    try:
        logger.info(f"Message: {request.message[:50]}...")
        
        # Call Gradio Space using gradio_client
        result = client.predict(
            message=request.message,
            history=request.history,
            api_name="/chat"  # ChatInterface default endpoint
        )
        
        logger.info(f"Response received: {str(result)[:50]}...")
        
        return ChatResponse(
            response=result,
            success=True,
            message="Response generated"
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
