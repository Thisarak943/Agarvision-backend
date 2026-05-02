from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .knowledge_base import compact_knowledge_text

load_dotenv()


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.last_error: str | None = None

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def extract(self, message: str, session_state: dict[str, Any]) -> dict[str, Any] | None:
        if not self.client:
            return None

        system_prompt = """
You are an information extraction layer for an agarwood oil market assistant.
Return ONLY valid JSON. Do not add markdown.

Supported intents:
- predict_demand
- explain_prediction
- price_recommendation
- current_prices
- competitor_info
- benefits_info
- oil_info
- market_info
- grade_info
- demand_index_info
- price_info
- festival_info
- help
- greeting
- reset
- unknown

Extract these fields when available:
oil_type, oil_grade, market_country, prediction_month, prediction_week, festival_season, language.

Rules:
- language must be one of: english, sinhala, mixed.
- prediction_month can be an English month name.
- If the user says "Month 12" or "M12", set prediction_month to 12.
- prediction_week must be an integer from 1 to 4 when available.
- If the user asks for demand with a specific oil, country, month, or week, use predict_demand.
- Use demand_index_info only for conceptual questions like "explain demand" or "what does demand index mean".
- festival_season must be true, false, or null.
- Use null for missing fields.
- For Sinhala-English mixed text, normalize known values to English where possible.
"""
        user_prompt = {
            "message": message,
            "current_session_state": session_state,
            "known_oils": [
                "Silani Ravana",
                "Silani Savera",
                "Silani Cobra",
                "Silani Junglefowl",
                "Silani Butterfly",
                "Silani Peacock",
                "Ravana",
                "Savera",
                "Cobra",
                "Junglefowl",
                "Butterfly",
                "Peacock",
            ],
            "known_grades": ["Premium", "Standard", "Budget"],
            "known_countries": [
                "UAE",
                "Saudi Arabia",
                "China",
                "Taiwan",
                "Japan",
                "Hong Kong",
                "Malaysia",
                "Indonesia",
                "Vietnam",
                "Singapore",
                "Spain",
                "Slovakia",
                "India",
                "Pakistan",
            ],
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content or "{}"
            self.last_error = None
            return json.loads(content)
        except Exception as error:
            self.last_error = str(error)
            return None

    def answer_knowledge(self, message: str, language: str) -> str | None:
        if not self.client:
            return None

        system_prompt = f"""
You are AgarVision's agarwood oil market assistant.
Answer using ONLY the knowledge base below.
If the answer is not in the knowledge base, say you do not have that detail.
Keep answers very short, farmer-friendly, and mobile-friendly.
Use at most 3 short sentences.
Prefer a natural conversational answer. Do not use bullet points unless the user asks for a list.
Use the user's language style. Requested language: {language}.

Knowledge base:
{compact_knowledge_text()}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=0.2,
            )
            self.last_error = None
            return response.choices[0].message.content
        except Exception as error:
            self.last_error = str(error)
            return None

    def health_check(self) -> dict[str, Any]:
        if not self.client:
            return {
                "configured": False,
                "model": self.model,
                "status": "missing_api_key",
                "error": "OPENAI_API_KEY is not configured.",
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Reply with OK only."},
                    {"role": "user", "content": "ping"},
                ],
                temperature=0,
                max_tokens=5,
            )
            return {
                "configured": True,
                "model": self.model,
                "status": "ok",
                "reply": response.choices[0].message.content,
            }
        except Exception as error:
            return {
                "configured": True,
                "model": self.model,
                "status": "error",
                "error": str(error),
            }
