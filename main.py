"""
FastAPI Backend for Qwen Chat API
Connects to your Hugging Face Space (Gradio-based)
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
# ✅ Use your actual Gradio Space endpoint, NOT the Inference API
HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://Redhanuman-qwen-chat-api.hf.space/api/predict/"
)

# Optional Hugging Face Token (only needed if your Space is private)
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Keep empty for public Space

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI(
    title="Qwen Chat Backend API",
    description="Backend API for Qwen 2.5 Chat Model via Hugging Face Space",
    version="2.0.0"
)

# Enable CORS (Frontend Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Request / Response Models
# -------------------------------------------------------------------
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
    """Check backend and Hugging Face Space connection"""
    try:
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        res = requests.post(
            HF_SPACE_URL,
            headers=headers,
            json={"data": ["ping", []]},
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
    Chat endpoint for sending messages to the Hugging Face Space.
    """
    try:
        logger.info(f"Incoming chat message: {request.message[:80]}")

        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        # 👇 This is how Gradio expects data
        payload = {
            "data": [request.message, request.history]
        }

        response = requests.post(
            HF_SPACE_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"Hugging Face Space error: {response.status_code}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"HF Space returned error: {response.text}"
            )

        result = response.json()
        logger.info(f"HF Space raw result: {result}")

        # Extract AI response
        ai_response = None
        if "data" in result and len(result["data"]) > 0:
            ai_response = result["data"][0]

        if not ai_response:
            raise HTTPException(
                status_code=500,
                detail="Empty response from Hugging Face Space"
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
        "model": "Redhanuman/qwen-chat-api (Gradio Space)",
        "backend": "FastAPI",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Gradio Space integration",
            "Conversation history",
            "Temperature control",
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
