from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from deployment.api.inference import ChatbotInference
from config import DEVICE


class ChatInput(BaseModel):
    user_input: str



app = FastAPI(title="Seq2Seq Chatbot API", version="1.0")


chatbot = None


@app.on_event("startup")
async def load_model():

    global chatbot
    model_path = "save/model/best_model.pth"

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")

        return


    chatbot = ChatbotInference(model_path, device=DEVICE)
    print(f"Model loaded successfully on {DEVICE}")



@app.post("/chat")
async def chat_endpoint(data: ChatInput):

    if chatbot is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:

        response_text = chatbot.evaluate(data.user_input)

        return {
            "input": data.user_input,
            "response": response_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

