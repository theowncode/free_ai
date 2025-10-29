"""
FastAPI Backend for Qwen Chat API
Connects to your Hugging Face Space
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qwen Chat Backend API",
    description="Backend API for Qwen 2.5 Chat Model",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your Hugging Face Space URL
HF_SPACE_URL = "https://redhanuman-qwen-chat-api.hf.space/api/predict"

# Request Models
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

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "status": "online",
        "message": "Qwen Chat Backend API is running",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    """Check if the API and HF Space are working"""
    try:
        # Test connection to HF Space
        response = requests.post(
            HF_SPACE_URL,
            json={"data": ["test", []]},
            timeout=30
        )
        
        if response.status_code == 200:
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
                "error": f"Status code: {response.status_code}"
            }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "backend": "online",
            "hf_space": "unreachable",
            "error": str(e)
        }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Example request:
    {
        "message": "Hello! Who are you?",
        "history": [["Hi", "Hello! How can I help?"]],
        "max_length": 512,
        "temperature": 0.7
    }
    """
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")
        
        # Prepare request for HF Space
        hf_request = {
            "data": [
                request.message,
                request.history
            ]
        }
        
        # Call Hugging Face Space API
        response = requests.post(
            HF_SPACE_URL,
            json=hf_request,
            timeout=60  # 60 second timeout
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get('data', [None])[0]
            
            if ai_response:
                logger.info("Chat response successful")
                return ChatResponse(
                    response=ai_response,
                    success=True,
                    message="Response generated successfully"
                )
            else:
                logger.error("Empty response from HF Space")
                raise HTTPException(
                    status_code=500,
                    detail="Received empty response from AI model"
                )
        else:
            logger.error(f"HF Space returned status: {response.status_code}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"AI model returned error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        raise HTTPException(
            status_code=504,
            detail="Request timeout - AI model took too long to respond"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to AI model: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint (for future implementation)
    Currently returns same as regular chat
    """
    return await chat(request)

@app.get("/api/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "model": "Qwen 2.5-1.5B-Instruct",
        "backend": "FastAPI",
        "hf_space": "redhanuman-qwen-chat-api",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "stats": "/api/stats"
        },
        "features": [
            "Conversation history support",
            "Configurable temperature",
            "Error handling",
            "CORS enabled",
            "Health monitoring"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "Please check the API documentation at /docs"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": "Something went wrong. Please try again later."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)