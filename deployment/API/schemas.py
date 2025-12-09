from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    message: str = Field(..., min_length=1, max_length=500, description="User input message")
    beam_size: int = Field(default=0, ge=0, le=10, description="Beam search width (0=greedy)")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "hello how are you?",
                "beam_size": 0
            }
        }


class ChatResponse(BaseModel):

    response: str = Field(..., description="Bot generated response")
    input: str = Field(..., description="Original user input")
    beam_size: int = Field(default=0, description="Beam size used")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "i am doing well thank you",
                "input": "hello how are you?",
                "beam_size": 0
            }
        }


class HealthResponse(BaseModel):

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used (cpu/cuda)")
