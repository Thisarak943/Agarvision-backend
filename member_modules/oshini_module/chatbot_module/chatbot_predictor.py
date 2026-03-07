from __future__ import annotations

import os
import pickle
import re
from typing import Dict, Any, Optional, List

import joblib
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import json
from pathlib import Path
from typing import Any, Dict

# openai imports
from openai import OpenAI


class ChatbotPredictor:
    
    def __init__(self, demand_predictor=None):

        base_dir = os.path.dirname(__file__)
        model_dir = os.path.join(base_dir, "model")
        
        self.chatbot_model_path = os.path.join(model_dir, "chatbot_model.pkl")
        self.vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        self.metadata_path = os.path.join(model_dir, "model_metadata.json")
        
        # init openai
        self._openai = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        context_path = Path(__file__).resolve().parents[1] / "context" / "oil_info_context.json"
        # If this file is located relative to THIS python file differently, adjust the parents[] count accordingly.
        self._oil_info_context: Dict[str, Any] = json.loads(context_path.read_text(encoding="utf-8"))

        # NLP components
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Integration with demand prediction
        self.demand_predictor = demand_predictor
        
        # Knowledge base
        self.oil_types = [
            "Silani Ravana", "Silani Savera", "Silani Cobra",
            "Silani Junglefowl", "Silani Butterfly", "Silani Peacock"
        ]
        self.oil_grades = ["Premium", "Standard", "Budget"]
        self.regions = ["Middle East", "East Asia", "Southeast Asia", "Europe", "South Asia"]
        self.countries = {
            "Middle East": ["UAE", "Saudi Arabia"],
            "East Asia": ["China", "Taiwan", "Japan"],
            "Southeast Asia": ["Singapore", "Malaysia"],
            "Europe": ["Spain", "Slovakia"],
            "South Asia": ["India"]
        }
        self.periods = ["Next Week", "Next Month", "Next Quarter"]
        
        # Conversation state (can be expanded for multi-turn)
        self.state = {
            "oil_type": None,
            "oil_grade": "Standard",  # default
            "market_region": None,
            "market_country": None,
            "prediction_period": None,
            "festival_season": False,
        }
        
        self._load()
    
    def _load(self):
        self.model = joblib.load(self.chatbot_model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        print("Chatbot model loaded successfully")
    
    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word.isalnum()]
        return ' '.join(words)
    
    def _predict_intent(self, message: str) -> tuple[str, float]:
        processed = self._preprocess_text(message)
        X = self.vectorizer.transform([processed])
        
        intent = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X).max()
        
        return intent, float(confidence)
    
    def _extract_oil_type(self, message: str) -> Optional[str]:
        message_lower = message.lower()
        
        # Check for each oil type
        for oil in self.oil_types:
            oil_lower = oil.lower()
            # Check full name or short name
            if oil_lower in message_lower or oil.split()[-1].lower() in message_lower:
                return oil
        
        return None
    
    def _extract_country(self, message: str) -> Optional[str]:
        message_upper = message.upper()
        
        # Flatten all countries
        all_countries = []
        for region_countries in self.countries.values():
            all_countries.extend(region_countries)
        
        for country in all_countries:
            if country.upper() in message_upper:
                return country
        
        return None
    
    def _extract_period(self, message: str) -> Optional[str]:
        message_lower = message.lower()
        
        if "week" in message_lower:
            return "Next Week"
        elif "quarter" in message_lower:
            return "Next Quarter"
        elif "month" in message_lower:
            return "Next Month"
        
        return None
    
    def _extract_oil_grade(self, message: str) -> Optional[str]:
        message_lower = message.lower()
        
        for grade in self.oil_grades:
            if grade.lower() in message_lower:
                return grade
        
        return None
    
    def _find_region_for_country(self, country: str) -> Optional[str]:
        for region, countries in self.countries.items():
            if country in countries:
                return region
        return None
    
    def _extract_entities(self, message: str, intent: str):
        # Oil type
        oil = self._extract_oil_type(message)
        if oil:
            self.state["oil_type"] = oil
        
        # Country
        country = self._extract_country(message)
        if country:
            self.state["market_country"] = country
            self.state["market_region"] = self._find_region_for_country(country)
        
        # Period
        period = self._extract_period(message)
        if period:
            self.state["prediction_period"] = period
        
        # Grade
        grade = self._extract_oil_grade(message)
        if grade:
            self.state["oil_grade"] = grade
        
        # Festival season detection
        if "festival" in message.lower():
            self.state["festival_season"] = True
    
    def _check_missing_parameters(self) -> List[str]:
        missing = []
        
        if not self.state["oil_type"]:
            missing.append("oil type (e.g., Ravana, Savera, Cobra)")
        if not self.state["market_country"]:
            missing.append("market country (e.g., UAE, China, Japan)")
        if not self.state["prediction_period"]:
            missing.append("prediction period (Next Week, Next Month, or Next Quarter)")
        
        return missing
    
    def _reset_state(self):
        self.state = {
            "oil_type": None,
            "oil_grade": "Standard",
            "market_region": None,
            "market_country": None,
            "prediction_period": None,
            "festival_season": False,
        }
    
    def _handle_greeting(self) -> Dict[str, Any]:
        return {
            "intent": "greeting",
            "response": "Hello! I'm your Agarwood Market Assistant. I can help you predict demand, recommend prices, and provide market insights. What would you like to know?"
        }
    
    def _handle_options(self) -> Dict[str, Any]:
        return {
            "intent": "options",
            "response": (
                "I can help you with:\n"
                "• Predict market demand for agarwood oils\n"
                "• Recommend pricing based on demand\n"
                "• Explain market factors and predictions\n"
                "• Provide information about oil types and markets\n\n"
                "Just tell me what you need!"
            )
        }
    
    def _handle_theory_demand(self) -> Dict[str, Any]:
        return {
            "intent": "theory_demand",
            "response": (
                "Demand represents the expected market interest for agarwood oil "
                "in a specific country and time period, measured on a 0-100 index scale. "
                "Higher index means stronger market demand."
            )
        }
    
    def _handle_theory_price(self) -> Dict[str, Any]:
        return {
            "intent": "theory_recommended_price",
            "response": (
                "Recommended price is a suggested price range based on predicted demand "
                "and current selling prices. Higher demand leads to higher recommended prices, "
                "while lower demand suggests competitive pricing."
            )
        }
    
    def _handle_oil_info(self, message) -> Dict[str, Any]:
        # Turn your JSON context into a compact string to send the model
        context_text = json.dumps(self._oil_info_context, ensure_ascii=False, indent=2)

        # system_instructions = (
        #     "You are an agarwood assistant.\n"
        #     "Use ONLY the provided context JSON to answer.\n"
        #     "If the context does not contain the answer, say what is missing and ask one follow-up question.\n"
        #     "Be clear and structured. Prefer bullets for lists."
        # )

        system_instructions = (
            "You are an agarwood assistant.\n"
            "Use ONLY the provided context JSON to answer.\n"
            "Be concise and mobile-friendly.\n"
            "Limit the response to 5–7 short bullet points.\n"
            "Each bullet must be one short sentence.\n"
            "Do NOT repeat information.\n"
            "Avoid long descriptions, background stories, or technical details.\n"
            "If the context does not contain the answer, clearly say what is missing "
            "and ask ONE short follow-up question.\n"
        )

        response = self._openai.responses.create(
            model="o3-mini",  # choose the model you want
            input=[
                {
                    "role": "system",
                    "content": system_instructions
                },
                {
                    "role": "user",
                    "content": (
                        "CONTEXT (JSON):\n"
                        f"{context_text}\n\n"
                        "USER QUESTION:\n"
                        f"{message}"
                    )
                }
            ],
        )

        print("MODEL OUTPUT : ", response.output_text)

        # return {
        #     "intent": "oil_info",
        #     "response": (
        #         "Sri Lanka exports 6 main agarwood oil types:\n"
        #         "• Silani Ravana (Premium grade)\n"
        #         "• Silani Savera (Premium grade)\n"
        #         "• Silani Cobra (Standard grade)\n"
        #         "• Silani Junglefowl (Standard grade)\n"
        #         "• Silani Butterfly (Standard grade)\n"
        #         "• Silani Peacock (Standard grade)\n\n"
        #         "Ravana and Savera are premium grades with higher market positioning."
        #     )
        # }
        return {
            "intent": "oil_info",
            "response": response.output_text
        }
    
    def _handle_market_info(self) -> Dict[str, Any]:
        return {
            "intent": "market_info",
            "response": (
                "Key export markets for Sri Lankan agarwood:\n"
                "• Middle East: UAE, Saudi Arabia\n"
                "• East Asia: China, Taiwan, Japan\n"
                "• Southeast Asia: Singapore, Malaysia\n"
                "• Europe: Spain, Slovakia\n"
                "• South Asia: India\n\n"
                "Each market has unique demand patterns and pricing dynamics."
            )
        }
    
    def _handle_prediction_periods(self) -> Dict[str, Any]:
        return {
            "intent": "prediction_periods",
            "response": (
                "I can forecast demand for three time horizons:\n"
                "• Next Week (7-day outlook)\n"
                "• Next Month (30-day outlook)\n"
                "• Next Quarter (90-day outlook)\n\n"
                "Which period would you like to analyze?"
            )
        }
    
    def _handle_missing_details(self) -> Dict[str, Any]:
        missing = self._check_missing_parameters()
        
        if missing:
            return {
                "intent": "missing_details",
                "response": (
                    f"To provide an accurate prediction, I need:\n"
                    f"{'• ' + chr(10) + '• '.join(missing)}\n\n"
                    f"Please provide these details."
                )
            }
        else:
            return {
                "intent": "missing_details",
                "response": "I have all the information needed! Would you like me to predict demand?"
            }
    
    def _handle_reset(self) -> Dict[str, Any]:
        self._reset_state()
        return {
            "intent": "reset",
            "response": "Conversation reset! Let's start fresh. What oil type would you like to analyze?"
        }
    
    def _handle_goodbye(self) -> Dict[str, Any]:
        self._reset_state()
        return {
            "intent": "goodbye",
            "response": "Goodbye! Good luck with your agarwood exports. Feel free to return anytime!"
        }
    
    def _handle_thank_you(self) -> Dict[str, Any]:
        return {
            "intent": "thank_you",
            "response": "You're welcome! Happy to help with your market analysis. Anything else?"
        }
    
    def _handle_fallback(self) -> Dict[str, Any]:
        return {
            "intent": "fallback",
            "response": (
                "I'm specialized in agarwood market analysis. I can help with demand prediction, "
                "pricing recommendations, and market information. Please ask something related to these areas."
            )
        }
    
    def _handle_noanswer(self) -> Dict[str, Any]:
        return {
            "intent": "noanswer",
            "response": "I didn't quite catch that. Could you rephrase your question about agarwood demand or pricing?"
        }
    
    def _handle_unsupported_selection(self) -> Dict[str, Any]:
        return {
            "intent": "unsupported_selection",
            "response": (
                "I currently support analysis for the 6 Silani oil types and markets in "
                "the Middle East, East Asia, Southeast Asia, Europe, and South Asia. "
                "The requested selection is not yet supported."
            )
        }
    
    def _handle_foreign_price_info(self) -> Dict[str, Any]:
        return {
            "intent": "foreign_price_info",
            "response": (
                "I provide Sri Lankan base selling prices and recommended price ranges "
                "based on demand forecasts. Country-specific market prices require "
                "external data sources, but I can predict demand for any supported market."
            )
        }
    
    def _handle_predict_demand(self, message: str) -> Dict[str, Any]:
        # Extract entities from message
        self._extract_entities(message, "predict_demand")
        
        # Check if we have all parameters
        missing = self._check_missing_parameters()
        
        if missing:
            return {
                "intent": "predict_demand",
                "status": "missing_parameters",
                "response": (
                    f"To predict demand, I need:\n"
                    f"{'• ' + chr(10) + '• '.join(missing)}\n\n"
                    f"Please provide these details."
                ),
                "missing_parameters": missing,
                "current_state": self.state.copy()
            }
        
        # Call demand predictor if available
        if not self.demand_predictor:
            return {
                "intent": "predict_demand",
                "status": "error",
                "response": "Demand prediction service is not available. Please contact support."
            }
        
        try:
            # Prepare payload for demand predictor
            payload = {
                "oil_type": self.state["oil_type"],
                "oil_grade": self.state["oil_grade"],
                "market_region": self.state["market_region"],
                "market_country": self.state["market_country"],
                "prediction_period": self.state["prediction_period"],
                "festival_season": self.state["festival_season"]
            }
            
            # Get prediction
            prediction = self.demand_predictor.predict(payload)
            
            # Format conversational response
            response = (
                f"Market Analysis for {self.state['oil_type']} in {self.state['market_country']}\n\n"
                f"Demand Index: {prediction['demand_index']}/100\n"
                f"Demand Level: {prediction['demand_level']}\n"
                f"Recommended Price Range: LKR {prediction['recommended_price_range']['min_price_lkr']:,} - "
                f"LKR {prediction['recommended_price_range']['max_price_lkr']:,}\n\n"
                f"Why this forecast:\n"
            )
            
            for reason in prediction["why"]:
                response += f"• {reason}\n"
            
            return {
                "intent": "predict_demand",
                "status": "success",
                "response": response,
                "prediction_data": prediction,
                "parameters_used": self.state.copy()
            }
            
        except Exception as e:
            return {
                "intent": "predict_demand",
                "status": "error",
                "response": f"Sorry, prediction failed: {str(e)}",
                "error": str(e)
            }
    
    def _handle_price_recommendation(self, message: str) -> Dict[str, Any]:
        # Extract entities
        self._extract_entities(message, "price_recommendation")
        
        # Check parameters
        missing = self._check_missing_parameters()
        
        if missing:
            return {
                "intent": "price_recommendation",
                "status": "missing_parameters",
                "response": (
                    f"To recommend pricing, I need:\n"
                    f"{'• ' + chr(10) + '• '.join(missing)}\n\n"
                    f"Please provide these details."
                ),
                "missing_parameters": missing
            }
        
        # Get prediction (includes pricing)
        if not self.demand_predictor:
            return {
                "intent": "price_recommendation",
                "status": "error",
                "response": "Pricing service is not available."
            }
        
        try:
            payload = {
                "oil_type": self.state["oil_type"],
                "oil_grade": self.state["oil_grade"],
                "market_region": self.state["market_region"],
                "market_country": self.state["market_country"],
                "prediction_period": self.state["prediction_period"],
                "festival_season": self.state["festival_season"]
            }
            
            prediction = self.demand_predictor.predict(payload)
            
            response = (
                f"Price Recommendation for {self.state['oil_type']} in {self.state['market_country']}\n\n"
                f"Based on demand forecast (Index: {prediction['demand_index']}, Level: {prediction['demand_level']}):\n\n"
                f"Recommended Price Range:\n"
                f"• Minimum: LKR {prediction['recommended_price_range']['min_price_lkr']:,}\n"
                f"• Maximum: LKR {prediction['recommended_price_range']['max_price_lkr']:,}\n\n"
                f"This pricing reflects current market conditions and predicted demand."
            )
            
            return {
                "intent": "price_recommendation",
                "status": "success",
                "response": response,
                "pricing_data": prediction["recommended_price_range"],
                "demand_context": {
                    "index": prediction["demand_index"],
                    "level": prediction["demand_level"]
                }
            }
            
        except Exception as e:
            return {
                "intent": "price_recommendation",
                "status": "error",
                "response": f"Sorry, pricing recommendation failed: {str(e)}"
            }
    
    def _handle_demand_and_price(self, message: str) -> Dict[str, Any]:
        return self._handle_predict_demand(message)  # Same as predict_demand
    
    def _handle_explain_xai(self, message: str) -> Dict[str, Any]:
        # Check if we have a recent prediction in state
        return {
            "intent": "explain_xai",
            "response": (
                "The prediction is based on multiple factors:\n"
                "• Seasonality (month, quarter, festival season)\n"
                "• Market region and country\n"
                "• Oil type and grade\n"
                "• Historical demand patterns\n"
                "• Market-specific economic indicators\n\n"
                "The model analyzes these factors to forecast demand on a 0-100 scale."
            )
        }
    
    def process_message(self, message: str) -> Dict[str, Any]:
        # Predict intent
        intent, confidence = self._predict_intent(message)
        
        # Route to appropriate handler
        handlers = {
            "greeting": self._handle_greeting,
            "options": self._handle_options,
            "theory_demand": self._handle_theory_demand,
            "theory_recommended_price": self._handle_theory_price,
            "predict_demand": lambda: self._handle_predict_demand(message),
            "price_recommendation": lambda: self._handle_price_recommendation(message),
            "demand_and_price": lambda: self._handle_demand_and_price(message),
            "explain_xai": lambda: self._handle_explain_xai(message),
            "oil_info": lambda: self._handle_oil_info(message),
            "market_info": self._handle_market_info,
            "reset": self._handle_reset,
            "goodbye": self._handle_goodbye,
            "thank_you": self._handle_thank_you,
            "fallback": self._handle_fallback,
            "noanswer": self._handle_noanswer,
            "unsupported_selection": self._handle_unsupported_selection,
            "prediction_periods": self._handle_prediction_periods,
            "missing_details": self._handle_missing_details,
            "foreign_price_info": self._handle_foreign_price_info,
        }
        
        # Get handler or default to fallback
        handler = handlers.get(intent, self._handle_fallback)
        
        # Get response
        result = handler()
        
        # Add metadata
        result["confidence"] = confidence
        result["detected_intent"] = intent
        
        return result