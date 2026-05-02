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
- Use predict_demand only when the user asks for future demand for a specific oil/grade/country/month/week.
- Use demand_index_info for conceptual questions like "what is demand in agarwood",
  "explain demand", "what does demand index mean", or Sinhala questions asking what demand/illuma means.
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

    def answer_market_question(self, message: str, language: str, intent: str) -> str | None:
        if not self.client:
            return None

        system_prompt = f"""
You are AgarVision's agarwood oil export market assistant.
Answer the user's question directly and naturally.

Scope:
- You can answer about agarwood oil, oud/fragrance uses, Sri Lankan agarwood oil types,
  export markets, farmers/exporters, market demand, prices, competitors, oil grades,
  festival season, and market guidance.
- Use the out-of-scope reply only when detected intent is "unknown" and the question is
  unrelated to agarwood oil or export-market guidance. The out-of-scope reply is exactly:
  "I can help with agarwood oil demand, prices, export markets, oil types, and market guidance. Please ask an agarwood market related question."
- If detected intent is not "unknown", treat the question as in-scope and answer it using the
  knowledge base and safe general agarwood-market knowledge.

Safety rules:
- Do not invent demand predictions. If the user asks for future demand, say they should provide
  oil type, oil grade, export country, month, and week so the demand model can be used.
- Do not invent current Sri Lankan selling prices. Use only the prices in the knowledge base.
- If intent is oil_info and the user asks for oil details/types in general, mention all 6 Silani
  oil types briefly in a compact answer. Do not ask the user to specify one unless they ask about
  a specific oil. Do not include prices unless the user asks for prices.
- If intent is current_prices, it is okay to use a compact list because prices are easier to read that way.
- If intent is benefits_info, explain why agarwood oil is useful/valuable and how the system
  helps farmers or exporters plan market, timing, demand, and price decisions.
- Keep the answer mobile-friendly: 1 to 3 short sentences.
- Support the requested language style: {language}.
- Do not use markdown headings or bold text.
- Do not end by asking the user if they need more details.
- If language is sinhala, answer mainly in Sinhala script. Some domain words like agarwood oil,
  demand, export, Premium, Standard, Budget, and country names may stay in English.
- If language is mixed, Singlish is allowed.
- Do not mention that you are using a knowledge base or an API.

Detected intent: {intent}

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
                temperature=0.5,
                max_tokens=320,
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
