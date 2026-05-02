from __future__ import annotations

import re
from typing import Any

from member_modules.oshini_module.demand_module.constants import (
    COUNTRIES_BY_REGION,
    COUNTRY_TO_REGION,
    CURRENT_OIL_PRICES_LKR,
    MONTH_NAME_TO_NUMBER,
    OIL_GRADES,
    OIL_TYPE_ALIASES,
)
from member_modules.oshini_module.demand_module.predictor import DemandPredictor
from member_modules.oshini_module.demand_module.schemas import DemandRequest

from .knowledge_base import KNOWLEDGE_BASE
from .llm_client import LLMClient


DEFAULT_SESSION_STATE = {
    "oil_type": None,
    "oil_grade": None,
    "market_country": None,
    "prediction_month": None,
    "prediction_week": None,
    "festival_season": False,
    "last_prediction": None,
}


class MarketChatbotAssistant:
    def __init__(self):
        self.llm = LLMClient()
        self.demand_predictor = DemandPredictor()
        self.sessions: dict[str, dict[str, Any]] = {}

    def reset(self, session_id: str = "default") -> None:
        self.sessions[session_id] = DEFAULT_SESSION_STATE.copy()

    def process_message(self, message: str, session_id: str = "default") -> dict[str, Any]:
        session_id = session_id or "default"
        state = self.sessions.setdefault(session_id, DEFAULT_SESSION_STATE.copy())

        fallback_extraction = self._fallback_extract(message)
        llm_extraction = self.llm.extract(message, state) or {}
        extraction = self._merge_extractions(fallback_extraction, llm_extraction)

        intent = extraction.get("intent") or "unknown"
        language = extraction.get("language") or self._detect_language(message)

        if intent == "reset":
            self.reset(session_id)
            return {
                "intent": "reset",
                "language": language,
                "reply": self._reply_reset(language),
                "status": "success",
                "missing_fields": [],
                "extracted": {},
                "prediction": None,
            }

        self._update_state(state, extraction)

        if intent == "price_info" and state.get("last_prediction"):
            meaning_words = ["meaning", "kiyanne", "mokakda", "what is", "explain"]
            if not any(word in message.lower() for word in meaning_words):
                intent = "price_recommendation"

        if intent == "predict_demand":
            return self._handle_prediction(state, language, reply_mode="demand_only")

        if intent == "explain_prediction":
            return self._handle_explain_prediction(state, language)

        if intent == "price_recommendation":
            return self._handle_price_recommendation(state, language)

        if intent in {
            "current_prices",
            "competitor_info",
            "benefits_info",
            "oil_info",
            "market_info",
            "grade_info",
            "demand_index_info",
            "price_info",
            "festival_info",
            "help",
            "greeting",
            "thanks",
            "unknown",
        }:
            return self._handle_knowledge(message, intent, language)

        return self._handle_knowledge(message, "unknown", language)

    def _handle_prediction(
        self,
        state: dict[str, Any],
        language: str,
        reply_mode: str = "demand_only",
    ) -> dict[str, Any]:
        market_country = state.get("market_country")
        if market_country and market_country not in COUNTRY_TO_REGION:
            return {
                "intent": "predict_demand",
                "language": language,
                "reply": self._unsupported_market_reply(str(market_country), language),
                "status": "unsupported_market",
                "missing_fields": [],
                "extracted": state.copy(),
                "prediction": None,
            }
        missing_fields = self._missing_prediction_fields(state)
        if missing_fields:
            return {
                "intent": "predict_demand",
                "language": language,
                "reply": self._missing_fields_reply(missing_fields, language),
                "status": "missing_fields",
                "missing_fields": missing_fields,
                "extracted": state.copy(),
                "prediction": None,
            }

        try:
            request = DemandRequest(
                oil_type=state["oil_type"],
                oil_grade=state["oil_grade"],
                market_country=state["market_country"],
                prediction_month=state["prediction_month"],
                prediction_week=state["prediction_week"],
                festival_season=bool(state["festival_season"]),
            )
            prediction = self.demand_predictor.predict(request.model_dump())
            state["last_prediction"] = prediction
            if reply_mode == "price_only":
                reply = self._price_reply(prediction, language)
            elif reply_mode == "reasons_only":
                reply = self._reasons_reply(prediction, language)
            else:
                reply = self._demand_only_reply(prediction, language)
            return {
                "intent": "predict_demand",
                "language": language,
                "reply": reply,
                "status": "success",
                "missing_fields": [],
                "extracted": request.model_dump(),
                "prediction": prediction,
            }
        except Exception as error:
            return {
                "intent": "predict_demand",
                "language": language,
                "reply": self._error_reply(str(error), language),
                "status": "error",
                "missing_fields": [],
                "extracted": state.copy(),
                "prediction": None,
            }

    def _handle_explain_prediction(self, state: dict[str, Any], language: str) -> dict[str, Any]:
        prediction = state.get("last_prediction")
        if prediction:
            return {
                "intent": "explain_prediction",
                "language": language,
                "reply": self._reasons_reply(prediction, language),
                "status": "success",
                "missing_fields": [],
                "extracted": state.copy(),
                "prediction": prediction,
            }
        return self._handle_prediction(state, language, reply_mode="reasons_only")

    def _handle_price_recommendation(self, state: dict[str, Any], language: str) -> dict[str, Any]:
        prediction = state.get("last_prediction")
        if prediction:
            return {
                "intent": "price_recommendation",
                "language": language,
                "reply": self._price_reply(prediction, language),
                "status": "success",
                "missing_fields": [],
                "extracted": state.copy(),
                "prediction": prediction,
            }
        return self._handle_prediction(state, language, reply_mode="price_only")

    def _handle_knowledge(self, message: str, intent: str, language: str) -> dict[str, Any]:
        if intent == "greeting":
            reply = self._greeting_reply(language)
        elif intent == "thanks":
            reply = self._thanks_reply(language)
        else:
            reply = self.llm.answer_market_question(message, language, intent) or self._fallback_knowledge_reply(intent, language)

        return {
            "intent": intent,
            "language": language,
            "reply": reply,
            "status": "success",
            "missing_fields": [],
            "extracted": {},
            "prediction": None,
        }

    def _fallback_extract(self, message: str) -> dict[str, Any]:
        text = message.lower()
        extracted: dict[str, Any] = {
            "intent": self._detect_intent(text),
            "language": self._detect_language(message),
        }

        for alias, full_name in OIL_TYPE_ALIASES.items():
            if alias.lower() in text or full_name.lower() in text:
                extracted["oil_type"] = full_name
                break

        for grade in OIL_GRADES:
            if grade.lower() in text:
                extracted["oil_grade"] = grade
                break

        for country in COUNTRY_TO_REGION:
            if country.lower() in text:
                extracted["market_country"] = country
                break

        for month_name, month_number in MONTH_NAME_TO_NUMBER.items():
            if month_name in text:
                extracted["prediction_month"] = month_name.title()
                break
        if "prediction_month" not in extracted:
            month_number_match = re.search(r"\b(?:month|m)\s*(1[0-2]|[1-9])\b", text)
            if month_number_match:
                extracted["prediction_month"] = int(month_number_match.group(1))

        sinhala_months = {
            "janawari": "January",
            "januari": "January",
            "februari": "February",
            "march": "March",
            "april": "April",
            "may": "May",
            "june": "June",
            "july": "July",
            "august": "August",
            "september": "September",
            "october": "October",
            "november": "November",
            "december": "December",
        }
        for token, month_name in sinhala_months.items():
            if token in text:
                extracted["prediction_month"] = month_name
                break

        week_match = re.search(r"(?:week|w)\s*([1-4])", text)
        if not week_match:
            week_match = re.search(r"\b([1-4])\s*(?:weni|vana|st|nd|rd|th)?\s*(?:week|sathiya|sathiy|තිය)\b", text)
        if week_match:
            extracted["prediction_week"] = int(week_match.group(1))
        elif "palaweni" in text or "first week" in text:
            extracted["prediction_week"] = 1
        elif "deweni" in text or "second week" in text:
            extracted["prediction_week"] = 2
        elif "thunweni" in text or "third week" in text:
            extracted["prediction_week"] = 3
        elif "hatharaweni" in text or "fourth week" in text:
            extracted["prediction_week"] = 4

        festival_terms = ["festival", "utsawa", "උත්සව"]
        no_festival_terms = [
            "no festival",
            "not festival",
            "without festival",
            "non festival",
            "non-festival",
            "festival season nemei",
            "festival season newei",
        ]
        if any(term in text for term in no_festival_terms):
            extracted["festival_season"] = False
        elif any(term in text for term in festival_terms) or "උත්සව" in message:
            extracted["festival_season"] = True

        return extracted

    def _detect_intent(self, text: str) -> str:
        has_prediction_details = self._has_prediction_details(text)
        if any(word in text for word in ["මොනවද කරන්න", "කරන්න පුළුවන්", "උදව්", "භාවිත", "කොහොමද use", "මේ system"]):
            return "help"
        if any(word in text for word in ["හයි", "හලෝ", "ආයුබෝවන්"]):
            return "greeting"
        if any(word in text for word in ["මිල", "ගණන්", "විකුණුම් මිල"]):
            return "current_prices"
        if any(word in text for word in ["benefit", "benefits", "farmer", "farmers", "prayo", "why we use", "why use", "use this oil"]):
            return "benefits_info"
        if "agarwood oil" in text and any(word in text for word in ["valuable", "value", "use", "why", "important"]):
            return "benefits_info"
        if any(word in text for word in ["ඇයි", "උනේ", "වුනේ", "හේතුව", "හේතු"]):
            return "explain_prediction"
        if any(word in text for word in ["why", "reason", "reasons", "une ai", "ay une", "ai une", "ai e", "ay e", "mokada"]):
            return "explain_prediction"
        if "demand" in text and has_prediction_details:
            return "predict_demand"
        if "demand" in text and any(word in text for word in ["explain", "meaning", "simple", "kiyanne", "mokakda"]):
            return "demand_index_info"
        if any(word in text for word in ["current price", "current prices", "selling price", "dan price"]):
            return "current_prices"
        if any(word in text for word in ["competitor", "competition", "compete"]):
            return "competitor_info"
        if any(word in text for word in ["thank", "thanks", "thank you", "sthuthi", "istuti"]):
            return "thanks"
        if any(word in text for word in ["hi", "hello", "hey", "ayubowan"]):
            return "greeting"
        if any(word in text for word in ["reset", "clear", "start over"]):
            return "reset"
        if any(word in text for word in ["help", "what can you do", "kohomada use", "use karanne"]):
            return "help"
        if any(word in text for word in ["demand", "predict", "forecast"]):
            return "predict_demand"
        if any(word in text for word in ["recommend price", "recommended price", "price recommendation"]):
            return "price_recommendation"
        if any(word in text for word in ["price", "mila", "ganan"]):
            return "price_info"
        if any(word in text for word in ["export", "country", "market", "destination"]):
            return "market_info"
        if any(word in text for word in ["grade", "premium", "standard", "budget"]):
            return "grade_info"
        if any(word in text for word in ["festival", "season", "utsawa"]):
            return "festival_info"
        if any(word in text for word in ["oil", "agarwood", "ravana", "savera", "cobra"]):
            return "oil_info"
        return "unknown"

    def _has_prediction_details(self, text: str) -> bool:
        has_oil = any(alias.lower() in text or full_name.lower() in text for alias, full_name in OIL_TYPE_ALIASES.items())
        has_country = any(country.lower() in text for country in COUNTRY_TO_REGION)
        has_month = any(month_name in text for month_name in MONTH_NAME_TO_NUMBER)
        has_month = has_month or re.search(r"\b(?:month|m)\s*(1[0-2]|[1-9])\b", text) is not None
        has_week = re.search(r"(?:week|w)\s*([1-4])", text) is not None
        return has_oil or has_country or has_month or has_week

    def _detect_language(self, message: str) -> str:
        has_sinhala = any("\u0d80" <= char <= "\u0dff" for char in message)
        ascii_words = any(char.isascii() and char.isalpha() for char in message)
        sinhala_roman = any(
            token in message.lower()
            for token in ["eka", "walata", "kohomada", "kiyanna", "danna", "mila", "ganan"]
        )
        if has_sinhala:
            return "sinhala"
        if sinhala_roman:
            return "mixed"
        return "english"

    def _merge_extractions(self, fallback: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
        merged = fallback.copy()
        fallback_intent = fallback.get("intent")
        for key, value in llm.items():
            if value is not None and value != "":
                merged[key] = value
        if fallback_intent in {
            "explain_prediction",
            "price_recommendation",
            "demand_index_info",
            "predict_demand",
            "current_prices",
            "competitor_info",
            "benefits_info",
            "thanks",
            "help",
            "greeting",
        }:
            merged["intent"] = fallback_intent
        if fallback.get("language") in {"sinhala", "mixed"}:
            merged["language"] = fallback["language"]
        if "festival_season" in fallback:
            merged["festival_season"] = fallback["festival_season"]
        if merged.get("oil_type") in OIL_TYPE_ALIASES:
            merged["oil_type"] = OIL_TYPE_ALIASES[str(merged["oil_type"])]
        if merged.get("oil_type"):
            oil_text = str(merged["oil_type"]).lower()
            for alias, full_name in OIL_TYPE_ALIASES.items():
                if oil_text in {alias.lower(), full_name.lower()}:
                    merged["oil_type"] = full_name
                    break
        if merged.get("oil_grade"):
            grade_text = str(merged["oil_grade"]).lower()
            for grade in OIL_GRADES:
                if grade_text == grade.lower():
                    merged["oil_grade"] = grade
                    break
        if merged.get("market_country"):
            country_text = str(merged["market_country"]).lower()
            for country in COUNTRY_TO_REGION:
                if country_text == country.lower():
                    merged["market_country"] = country
                    break
        if merged.get("prediction_month") and not isinstance(merged["prediction_month"], int):
            month_text = str(merged["prediction_month"]).lower()
            for month_name in MONTH_NAME_TO_NUMBER:
                if month_text == month_name:
                    merged["prediction_month"] = month_name.title()
                    break
        return merged

    def _update_state(self, state: dict[str, Any], extraction: dict[str, Any]) -> None:
        for key in DEFAULT_SESSION_STATE:
            if key in extraction and extraction[key] is not None and key != "last_prediction":
                state[key] = extraction[key]

    def _missing_prediction_fields(self, state: dict[str, Any]) -> list[str]:
        required = ["oil_type", "oil_grade", "market_country", "prediction_month", "prediction_week"]
        return [field for field in required if not state.get(field)]

    def _missing_fields_reply(self, missing_fields: list[str], language: str) -> str:
        if len(missing_fields) == 1:
            field = missing_fields[0]
            if field == "oil_type":
                return self._single_missing_reply("oil type", "Silani Ravana", language)
            if field == "oil_grade":
                return self._single_missing_reply("oil grade", "Premium, Standard, or Budget", language)
            if field == "market_country":
                examples = ", ".join(list(COUNTRY_TO_REGION)[:6])
                return self._single_missing_reply("export country", examples, language)
            if field == "prediction_month":
                return self._single_missing_reply("prediction month", "January or Month 1", language)
            if field == "prediction_week":
                return self._single_missing_reply("prediction week", "Week 1, 2, 3, or 4", language)

        labels = {
            "oil_type": "oil type",
            "oil_grade": "oil grade",
            "market_country": "export country",
            "prediction_month": "prediction month",
            "prediction_week": "prediction week",
        }
        missing_text = ", ".join(labels[field] for field in missing_fields)
        if language == "sinhala":
            return f"Demand prediction එකට තව {missing_text} දෙන්න."
        if language == "mixed":
            return f"Demand eka predict karanna thawa {missing_text} denna."
        return f"To predict demand, please provide: {missing_text}."

    def _single_missing_reply(self, label: str, example: str, language: str) -> str:
        if language == "sinhala":
            return f"Demand prediction එකට {label} එක විතරක් දෙන්න. උදාහරණ: {example}."
        if language == "mixed":
            return f"Demand eka predict karanna {label} eka witharak denna. Example: {example}."
        return f"To predict demand, please provide only the {label}. Example: {example}."

    def _unsupported_market_reply(self, country: str, language: str) -> str:
        supported = ", ".join(COUNTRY_TO_REGION)
        if language == "sinhala":
            return (
                f"{country} දැනට supported export market එකක් නෙවෙයි. "
                f"කරුණාකර supported country එකක් තෝරන්න: {supported}."
            )
        if language == "mixed":
            return (
                f"{country} danata supported export market ekak newei. "
                f"Please supported country ekak select karanna: {supported}."
            )
        return (
            f"{country} is not a supported export market for this demand prediction. "
            f"Please select one of these countries: {supported}."
        )

    def _demand_only_reply(self, prediction: dict[str, Any], language: str) -> str:
        price = prediction["recommended_price_range"]
        main_reason = self._compact_reason(prediction)
        if language == "sinhala":
            sinhala_reason = self._compact_reason_sinhala(prediction)
            return (
                f"{prediction['oil_name']} {prediction['oil_grade']} තෙල් "
                f"{prediction['export_country']} වෙත {prediction['export_date']} අපනයනය කරන විට "
                f"ඉල්ලුම {prediction['demand_category']} මට්ටමේයි. "
                f"ඉල්ලුම් දර්ශකය {prediction['demand_index']}. {sinhala_reason} "
                f"නිර්දේශිත මිල පරාසය LKR {price['min_price_lkr']:,} - {price['max_price_lkr']:,}."
            )
        if language == "mixed":
            return (
                f"{prediction['oil_name']} {prediction['oil_grade']} oil eka "
                f"{prediction['export_country']} walata {prediction['export_date']} export karanna "
                f"predicted demand eka {prediction['demand_category']} level ekak. "
                f"Demand index eka {prediction['demand_index']}. {main_reason} "
                f"Recommended price range eka LKR {price['min_price_lkr']:,} - {price['max_price_lkr']:,}."
            )
        return (
            f"For {prediction['oil_name']} {prediction['oil_grade']} oil exported to "
            f"{prediction['export_country']} in {prediction['export_date']}, predicted demand is "
            f"{prediction['demand_category']} with demand index {prediction['demand_index']}. "
            f"{main_reason} Recommended price range is LKR {price['min_price_lkr']:,} - {price['max_price_lkr']:,}."
        )

    def _compact_reason(self, prediction: dict[str, Any]) -> str:
        reasons = prediction.get("reasons", [])
        if not reasons:
            return "This result is based on the selected oil, market, and time period."
        first = reasons[0].rstrip(".")
        festival_reason = next(
            (reason.rstrip(".") for reason in reasons if "Festival season" in reason),
            "",
        )
        country_reason = next(
            (reason.rstrip(".") for reason in reasons if prediction["export_country"] in reason),
            "",
        )
        if festival_reason and festival_reason != first:
            return f"This is mainly because {self._lower_first(first)} and {self._lower_first(festival_reason)}."
        if country_reason and country_reason != first:
            return f"This is mainly because {self._lower_first(first)} and {self._lower_first(country_reason)}."
        return f"This is mainly because {self._lower_first(first)}."

    def _compact_reason_sinhala(self, prediction: dict[str, Any]) -> str:
        country = prediction["export_country"]
        grade = prediction["oil_grade"]
        month = prediction["predicted_time"]["month_name"]
        return (
            f"මෙයට ප්‍රධාන හේතුව {grade} grade එක ඉල්ලුමට බලපාන අතර "
            f"{country} වෙළඳපොළ agarwood oil සඳහා ශක්තිමත් වීමයි. "
            f"{month} කාලයත් මෙම ප්‍රතිඵලයට බලපායි."
        )

    def _lower_first(self, text: str) -> str:
        if not text:
            return text
        first_word = text.split(" ", 1)[0]
        if first_word.isupper() or first_word in COUNTRY_TO_REGION:
            return text
        return text[0].lower() + text[1:]

    def _reasons_reply(self, prediction: dict[str, Any], language: str) -> str:
        reason_text = " ".join(prediction["reasons"])
        if language == "sinhala":
            return (
                f"ඒක {prediction['demand_category']} වුණේ ප්‍රධාන වශයෙන් "
                f"{prediction['oil_grade']} grade එක ඉල්ලුමට බලපාන නිසා. "
                f"{prediction['export_country']} වෙළඳපොළ ශක්තිමත් වීමත් "
                f"{prediction['export_date']} කාලය demand එකට ගැලපීමත් මේ ප්‍රතිඵලයට හේතු වුණා."
            )
        if language == "mixed":
            return (
                f"Me prediction eka {prediction['demand_category']} une mainly me factors nisa. "
                f"{reason_text} Me factors okkoma ekathu unama {prediction['export_country']} market ekata "
                f"{prediction['export_date']} demand level eka me wage pennanawa."
            )
        return (
            f"This prediction is {prediction['demand_category']} mainly because of these factors: "
            f"{reason_text} Together, these conditions explain the demand level for "
            f"{prediction['export_country']} in {prediction['export_date']}."
        )

    def _price_reply(self, prediction: dict[str, Any], language: str) -> str:
        price = prediction["recommended_price_range"]
        if language == "sinhala":
            return (
                f"{prediction['oil_name']} {prediction['oil_grade']} තෙල් "
                f"{prediction['export_country']} වෙත {prediction['export_date']} අපනයනය කරනවා නම්, "
                f"නිර්දේශිත මිල පරාසය LKR {price['min_price_lkr']:,} සිට "
                f"LKR {price['max_price_lkr']:,} දක්වායි. "
                f"මෙය {prediction['demand_category']} demand level එක මත පදනම් වේ."
            )
        if language == "mixed":
            return (
                f"{prediction['oil_name']} {prediction['oil_grade']} oil eka "
                f"{prediction['export_country']} walata {prediction['export_date']} export karanawanam, "
                f"recommended price range eka LKR {price['min_price_lkr']:,} idan "
                f"LKR {price['max_price_lkr']:,} dakwa. Me range eka predicted demand "
                f"{prediction['demand_category']} level eka anuwa calculate karala thiyenne."
            )
        return (
            f"For {prediction['oil_name']} {prediction['oil_grade']} oil exported to "
            f"{prediction['export_country']} in {prediction['export_date']}, the recommended price range is "
            f"LKR {price['min_price_lkr']:,} to LKR {price['max_price_lkr']:,}. "
            f"This range is based on the predicted {prediction['demand_category']} demand level."
        )
    def _fallback_knowledge_reply(self, intent: str, language: str) -> str:
        if intent == "market_info":
            markets = "; ".join(
                f"{region}: {', '.join(countries)}"
                for region, countries in COUNTRIES_BY_REGION.items()
            )
            return f"Supported export markets are {markets}."
        if intent == "grade_info":
            return "Oil grades are Premium, Standard, and Budget. Premium is highest quality, Standard is regular export quality, and Budget is more price-sensitive."
        if intent == "demand_index_info":
            return KNOWLEDGE_BASE["demand_index"]
        if intent == "price_info":
            return KNOWLEDGE_BASE["recommended_price"]
        if intent == "current_prices":
            return self._current_prices_reply(language)
        if intent == "benefits_info":
            return self._benefits_reply(language)
        if intent == "competitor_info":
            return self._competitor_reply(language)
        if intent == "festival_info":
            return KNOWLEDGE_BASE["festival_season"]
        if intent == "oil_info":
            return self._oil_details_reply(language)
        return self._out_of_scope_reply(language)

    def _greeting_reply(self, language: str) -> str:
        if language == "sinhala":
            return (
                "හයි! මම AgarVision market assistant. "
                "Agarwood oil demand, prices, oil types, සහ export markets ගැන සරලව උදව් කරන්න පුළුවන්."
            )
        if language == "mixed":
            return (
                "Hi! Mama AgarVision market assistant. Oyata agarwood oil export market eka gana "
                "therum ganna, future demand eka balanna, oil types, current prices, export countries "
                "gana saralawa dana ganna help karanna puluwan."
            )
        return (
            "Hi! I am your AgarVision market assistant. I can help you understand agarwood oil export markets, "
            "check future demand, and learn about oil types, current prices, and export countries in a simple way."
        )

    def _thanks_reply(self, language: str) -> str:
        if language == "sinhala":
            return "ඔයාට සාදරයෙන්. Agarwood oil market ගැන තව දෙයක් දැනගන්න ඕන නම් මට කියන්න."
        if language == "mixed":
            return "Oyata welcome. Agarwood oil market gana thawath deyak ona nam mata kiyanna."
        return "You are welcome. Ask me anytime about agarwood oil demand, prices, or export markets."

    def _out_of_scope_reply(self, language: str) -> str:
        if language == "sinhala":
            return (
                "මට උදව් කරන්න පුළුවන් agarwood oil demand, prices, export markets, "
                "oil types, සහ market guidance ගැන විතරයි. කරුණාකර agarwood market related question එකක් අහන්න."
            )
        if language == "mixed":
            return (
                "Mata help karanna puluwan agarwood oil demand, prices, export markets, "
                "oil types, saha market guidance gana witharai. Please agarwood market related question ekak ahanna."
            )
        return (
            "I can help with agarwood oil demand, prices, export markets, oil types, and market guidance. "
            "Please ask an agarwood market related question."
        )

    def _help_reply(self, language: str) -> str:
        if language == "sinhala":
            return (
                "මේ feature එකෙන් agarwood oil export demand predict කරන්න, demand එක High/Medium/Low ද කියලා බලන්න, "
                "හේතු සහ recommended price range එක ගන්න පුළුවන්. Oil types, current prices, export countries, සහ competitors ගැනත් මට සරලව කියන්න පුළුවන්."
            )
        if language == "mixed":
            return (
                "Me feature eken oyata agarwood oil export decision ganna udaw wenawa. Oil type, grade, "
                "export country, month, week dunnama future demand index eka saha demand category eka balanna puluwan. "
                "Passe oyata one nam demand eka ehema une ai kiyala reasons ahanna puluwan, price recommendation ekath "
                "wenama ahanna puluwan. Oil types, current prices, export countries, competitors wage market detailsuth "
                "mama saralawa explain karannam."
            )
        return (
            "This feature helps you make better agarwood oil export decisions. If you give me an oil type, grade, "
            "export country, month, and week, I can check the future demand index and demand category. After that, "
            "you can ask why that demand level was predicted or ask for a recommended price range. I can also explain "
            "oil types, current prices, export countries, competitors, and market details in simple language."
        )

    def _current_prices_reply(self, language: str) -> str:
        prices = ", ".join(
            f"{oil} LKR {price:,}"
            for oil, price in CURRENT_OIL_PRICES_LKR.items()
        )
        if language == "sinhala":
            return f"Sri Lanka agarwood oil වල current selling prices මෙහෙමයි: {prices}."
        if language == "mixed":
            return f"Sri Lanka wala current agarwood oil selling prices mehema thiyenawa: {prices}."
        return f"The current Sri Lankan agarwood oil selling prices are: {prices}."

    def _oil_details_reply(self, language: str) -> str:
        if language == "sinhala":
            return (
                "Sri Lanka export කරන Silani oil types 6ක් තියෙනවා: Ravana සහ Savera premium/luxury buyers සඳහා හොඳයි; "
                "Cobra price-sensitive buyers සඳහා ගැලපෙනවා; Junglefowl, Butterfly, Peacock regular export demand සඳහා stable options. "
                "මේ oils fragrance, perfume, oud products, religious/cultural use, සහ luxury gifting markets වලට යොදාගන්නවා."
            )
        if language == "mixed":
            return (
                "Sri Lanka export karana Silani oil types 6k thiyenawa: Ravana saha Savera premium/luxury buyers lata hondai; "
                "Cobra price-sensitive buyers lata galapenawa; Junglefowl, Butterfly, Peacock regular export demand walata stable options. "
                "Me oils fragrance, perfume, oud products, religious/cultural use, saha luxury gifting markets walata use wenawa."
            )
        return (
            "Sri Lanka exports 6 Silani agarwood oil types: Ravana and Savera are premium oils for luxury buyers; "
            "Cobra is more suitable for price-sensitive buyers; Junglefowl, Butterfly, and Peacock are stable regular export options. "
            "These oils are mainly used for fragrance, perfume, oud products, religious or cultural use, and luxury gifting markets."
        )

    def _benefits_reply(self, language: str) -> str:
        if language == "sinhala":
            return (
                "Agarwood oil fragrance, perfume, oud products, religious/cultural use, සහ luxury gifting සඳහා වටිනා oil එකක්. "
                "Exporters සහ farmers ලාට demand prediction එකෙන් හොඳ market එක, හොඳ time එක, සහ suitable price range එක plan කරන්න පුළුවන්."
            )
        if language == "mixed":
            return (
                "Agarwood oil eka fragrance, perfume, oud products, religious/cultural use, saha luxury gifting walata valuable. "
                "Exporters saha farmers lata demand prediction eken hoda market eka, hoda time eka, saha suitable price range eka plan karanna puluwan."
            )
        return (
            "Agarwood oil is valuable for fragrance, perfume, oud products, religious or cultural use, and luxury gifting. "
            "For exporters and farmers, demand prediction helps choose a better market, selling time, and suitable price range."
        )

    def _competitor_reply(self, language: str) -> str:
        if language == "sinhala":
            return (
                "Sri Lankan agarwood oil exporters ලාට ප්‍රධාන competition එක Malaysia, Indonesia, Vietnam වගේ Southeast Asian suppliers ලාගෙන් එනවා. "
                "Competition එක price විතරක් නෙවෙයි; oil quality, buyer trust, certification, සහ export reliabilityත් වැදගත්."
            )
        if language == "mixed":
            return (
                "Sri Lankan agarwood oil exporters lata mainly competition enne Southeast Asia region eke established "
                "agarwood and oud suppliers lagen, especially Malaysia, Indonesia, Vietnam wage markets walin. Competition "
                "eka price eka witharak newei; oil quality, consistency, buyer trust, certification, saha export reliability "
                "wage dewaluth balapanawa."
            )
        return (
            "Sri Lankan agarwood oil exporters usually compete with established agarwood and oud suppliers from Southeast "
            "Asian markets such as Malaysia, Indonesia, and Vietnam. Competition is not only about price; oil quality, "
            "consistency, buyer trust, certification, and export reliability also matter."
        )
    def _reply_reset(self, language: str) -> str:
        if language == "sinhala":
            return "Conversation එක reset කළා. දැන් අලුත් prediction එකක් අහන්න පුළුවන්."
        if language == "mixed":
            return "Conversation eka reset kala. Aluth prediction ekak ahanna puluwan."
        return "Conversation reset. You can ask for a new prediction."

    def _error_reply(self, error: str, language: str) -> str:
        if language == "sinhala":
            return f"Sorry, prediction එක complete කරන්න බැරි වුණා. Error: {error}"
        if language == "mixed":
            return f"Sorry, prediction eka complete karanna bari una. Error: {error}"
        return f"Sorry, I could not complete the prediction. Error: {error}"
