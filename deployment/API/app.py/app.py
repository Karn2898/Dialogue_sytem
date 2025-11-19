
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from deployment.api.inference import ChatbotInference
from config import DEVICE

app = FastAPI(title="Seq2Seq Chatbot API", version="1.0")

# Initialize model at startup
chatbot = None


@app.on_event("startup")
async def load_model():
    """Load model on server startup"""
    global chatbot
    model_path = "save/model/best_model.pth"
    chatbot = ChatbotInference(model_path, device=DEVICE)
    print(f"Model loaded successfully from {model_path}")


class ChatRequest(BaseModel):
    message: str
    beam_size: int = 0  # 0 for greedy, >0 for beam search


class ChatResponse(BaseModel):
    response: str
    input: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
   #generate chatbot response
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        response = chatbot.generate_response(
            request.message,
            beam_size=request.beam_size
        )
        return ChatResponse(response=response, input=request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": chatbot is not None
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
