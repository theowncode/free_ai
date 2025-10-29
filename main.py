"""
FastAPI Backend for Qwen Chat API
Connects to your Hugging Face Model via Inference API
Run locally:  uvicorn main:app --reload
Deployable on Render or any FastAPI-compatible host
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import logging

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen-backend")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://api-inference.huggingface.co/models/Redhanuman/qwen-chat-api"
)
HF_TOKEN = os.getenv("HF_TOKEN", "hf_yourRealTokenHere")

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI(
    title="Qwen Chat Backend API",
    description="Backend API for Qwen 2.5 Chat Model using Hugging Face Inference API",
    version="1.1.0"
)

# CORS (frontend can connect from any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Request / Response Models
# -------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[List[str]]] = []
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    success: bool
    message: Optional[str] = None

# -------------------------------------------------------------------
# Root Endpoint
# -------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Qwen Chat Backend API is running",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

# -------------------------------------------------------------------
# Health Check
# -------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Check backend and HF API connection"""
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        res = requests.post(
            HF_SPACE_URL,
            headers=headers,
            json={"inputs": "test"},
            timeout=10
        )

        if res.status_code == 200:
            return {
                "status": "healthy",
                "backend": "online",
                "hf_space": "connected",
                "hf_space_url": HF_SPACE_URL
            }
        else:
            return {
                "status": "degraded",
                "backend": "online",
                "hf_space": "error",
                "error": f"Status code: {res.status_code}",
                "text": res.text
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "backend": "online",
            "hf_space": "unreachable",
            "error": str(e)
        }

# -------------------------------------------------------------------
# Main Chat Endpoint
# -------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint to send messages to the Qwen model
    Example request:
    {
        "message": "Hello Qwen!",
        "history": [["Hi", "Hello! How can I help?"]],
        "max_length": 512,
        "temperature": 0.7
    }
    """
    try:
        logger.info(f"Received chat message: {request.message[:80]}")

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": request.message,
            "parameters": {
                "max_new_tokens": request.max_length,
                "temperature": request.temperature
            }
        }

        response = requests.post(
            HF_SPACE_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"HF API returned error: {response.status_code}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Hugging Face API error: {response.text}"
            )

        result = response.json()
        logger.info(f"HF API raw output: {result}")

        ai_response = ""
        if isinstance(result, list) and len(result) > 0:
            ai_response = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            ai_response = result.get("generated_text", "")

        if not ai_response:
            raise HTTPException(
                status_code=500,
                detail="Empty response from Hugging Face model"
            )

        return ChatResponse(
            response=ai_response.strip(),
            success=True,
            message="Response generated successfully"
        )

    except requests.exceptions.Timeout:
        logger.error("Timeout while waiting for Hugging Face response")
        raise HTTPException(status_code=504, detail="Model timeout")
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Stats Endpoint
# -------------------------------------------------------------------
@app.get("/api/stats")
async def get_stats():
    return {
        "model": "Redhanuman/qwen-chat-api",
        "backend": "FastAPI",
        "version": "1.1.0",
        "status": "operational",
        "features": [
            "Conversation history",
            "Temperature control",
            "Error handling",
            "Health monitoring",
            "CORS enabled"
        ]
    }

# -------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "See API docs at /docs"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": str(exc)
    }

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
