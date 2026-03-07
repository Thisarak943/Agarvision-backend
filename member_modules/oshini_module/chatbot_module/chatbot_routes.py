from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import both predictors
from member_modules.oshini_module.demand_module.predictor import DemandPredictor
from .chatbot_predictor import ChatbotPredictor

router = APIRouter(prefix="/chatbot", tags=["Chatbot - Conversational AI"])

# Initialize predictors
demand_predictor = DemandPredictor()
chatbot = ChatbotPredictor(demand_predictor=demand_predictor)


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None  # For future multi-user session management


class ChatResponse(BaseModel):
    intent: str
    response: str
    confidence: float
    status: Optional[str] = "success"
    data: Optional[dict] = None


@router.get("/health") 
def health():
    return {
        "status": "ok",
        "service": "chatbot",
        "model_loaded": chatbot.model is not None,
        "demand_integration": chatbot.demand_predictor is not None
    }


@router.post("/chat", response_model=ChatResponse)
def chat(msg: ChatMessage):

    try:
        # Validate input
        if not msg.message or not msg.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        # Process message through chatbot
        result = chatbot.process_message(msg.message.strip())
        
        # Prepare response
        response = ChatResponse(
            intent=result["detected_intent"],
            response=result["response"],
            confidence=result["confidence"],
            status=result.get("status", "success"),
            data=result.get("prediction_data") or result.get("pricing_data")
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chatbot error: {str(e)}"
        )


@router.post("/reset-conversation")
def reset_conversation():
    chatbot._reset_state()
    return {
        "status": "ok",
        "message": "Conversation reset successfully"
    }


@router.get("/supported-intents")
def get_supported_intents():
    return {
        "intents": [
            {
                "name": "greeting",
                "description": "Greet the chatbot",
                "examples": ["Hello", "Hi", "Good morning"]
            },
            {
                "name": "options",
                "description": "Ask what chatbot can do",
                "examples": ["What can you do?", "Help", "How can you assist?"]
            },
            {
                "name": "predict_demand",
                "description": "Predict market demand",
                "examples": [
                    "Predict demand for Ravana in UAE next month",
                    "What is the demand for Cobra in China?",
                    "Forecast demand for Savera"
                ]
            },
            {
                "name": "price_recommendation",
                "description": "Get pricing recommendations",
                "examples": [
                    "Recommend price for Ravana",
                    "What price should I charge for Cobra in UAE?",
                    "Pricing for Peacock next quarter"
                ]
            },
            {
                "name": "explain_xai",
                "description": "Explain prediction factors",
                "examples": [
                    "Why this prediction?",
                    "Explain the forecast",
                    "What factors were considered?"
                ]
            },
            {
                "name": "oil_info",
                "description": "Information about oil types",
                "examples": [
                    "Tell me about the 6 oil types",
                    "What is Silani Ravana?",
                    "List agarwood oils"
                ]
            },
            {
                "name": "market_info",
                "description": "Information about export markets",
                "examples": [
                    "Where can I export?",
                    "Key export destinations",
                    "Tell me about Middle East market"
                ]
            },
            {
                "name": "reset",
                "description": "Reset conversation",
                "examples": ["Reset", "Start over", "New conversation"]
            },
            {
                "name": "goodbye",
                "description": "End conversation",
                "examples": ["Goodbye", "Bye", "See you"]
            }
        ]
    }


@router.get("/knowledge-base")
def get_knowledge_base():
    return {
        "oil_types": chatbot.oil_types,
        "oil_grades": chatbot.oil_grades,
        "regions": chatbot.regions,
        "countries": chatbot.countries,
        "prediction_periods": chatbot.periods
    }


@router.post("/test-intent-classification")
def test_intent_classification(msg: ChatMessage):
    try:
        intent, confidence = chatbot._predict_intent(msg.message)
        
        return {
            "message": msg.message,
            "predicted_intent": intent,
            "confidence": confidence,
            "threshold_met": confidence > 0.5
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )