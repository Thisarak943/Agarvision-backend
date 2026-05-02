from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .assistant import MarketChatbotAssistant
from .schemas import ChatRequest, ChatResponse, ResetRequest

router = APIRouter(prefix="/oshini/market-chatbot", tags=["Oshini - Market Chatbot"])

assistant = MarketChatbotAssistant()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "market_chatbot",
        "llm_configured": assistant.llm.is_configured,
        "llm_model": assistant.llm.model,
        "demand_integration": assistant.demand_predictor is not None,
    }


@router.get("/llm-health")
def llm_health():
    return assistant.llm.health_check()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return assistant.process_message(
            message=request.message.strip(),
            session_id=request.session_id or "default",
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {error}")


@router.post("/reset")
def reset(request: ResetRequest):
    assistant.reset(request.session_id or "default")
    return {"status": "ok", "message": "Conversation reset successfully."}
