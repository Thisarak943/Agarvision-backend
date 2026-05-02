from __future__ import annotations

from member_modules.oshini_module.demand_module.constants import (
    COUNTRIES_BY_REGION,
    CURRENT_OIL_PRICES_LKR,
)


KNOWLEDGE_BASE = {
    "assistant_scope": (
        "AgarVision market assistant helps users understand agarwood oil export demand, "
        "oil types, export countries, demand categories, and recommended price ranges."
    ),
    "oil_types": {
        "Silani Ravana": {
            "positioning": "High-value premium agarwood oil for luxury export markets.",
            "current_price_lkr": CURRENT_OIL_PRICES_LKR["Silani Ravana"],
        },
        "Silani Savera": {
            "positioning": "High-value agarwood oil positioned for premium buyers.",
            "current_price_lkr": CURRENT_OIL_PRICES_LKR["Silani Savera"],
        },
        "Silani Cobra": {
            "positioning": "More price-sensitive oil type compared with premium oils.",
            "current_price_lkr": CURRENT_OIL_PRICES_LKR["Silani Cobra"],
        },
        "Silani Junglefowl": {
            "positioning": "Stable standard-market agarwood oil.",
            "current_price_lkr": CURRENT_OIL_PRICES_LKR["Silani Junglefowl"],
        },
        "Silani Butterfly": {
            "positioning": "Stable regular export-market agarwood oil.",
            "current_price_lkr": CURRENT_OIL_PRICES_LKR["Silani Butterfly"],
        },
        "Silani Peacock": {
            "positioning": "Stable regular export-market agarwood oil.",
            "current_price_lkr": CURRENT_OIL_PRICES_LKR["Silani Peacock"],
        },
    },
    "oil_grades": {
        "Premium": "Highest quality grade, usually better for premium export buyers.",
        "Standard": "Regular export grade with stable market demand.",
        "Budget": "Lower-price grade, usually more price-sensitive.",
    },
    "export_markets": COUNTRIES_BY_REGION,
    "demand_index": (
        "Demand index is a 0-100 score showing expected market demand. "
        "Low is 45 or below, Medium is above 45 up to 65, and High is above 65."
    ),
    "recommended_price": (
        "Recommended price range is calculated from the current Sri Lankan selling price "
        "and the predicted demand category. High demand can support a higher range, "
        "while low demand suggests a more competitive range."
    ),
    "festival_season": (
        "Festival season can increase buyer interest because agarwood oil is linked with "
        "fragrance, cultural, and luxury use cases in some export markets."
    ),
    "benefits": (
        "The feature helps growers and exporters understand future demand before choosing "
        "a market. It can show the expected demand level, explain the main reasons, and "
        "suggest a price range based on current Sri Lankan selling prices."
    ),
    "competitors": (
        "Sri Lankan agarwood oil exporters usually compete with established agarwood and oud "
        "suppliers from Southeast Asian markets such as Malaysia, Indonesia, Vietnam, and "
        "other regional trading hubs. Competition can depend on oil quality, buyer trust, "
        "certification, consistency, and price."
    ),
}


def compact_knowledge_text() -> str:
    oil_lines = [
        f"- {name}: {details['positioning']} Current price LKR {details['current_price_lkr']:,}."
        for name, details in KNOWLEDGE_BASE["oil_types"].items()
    ]
    market_lines = [
        f"- {region}: {', '.join(countries)}"
        for region, countries in COUNTRIES_BY_REGION.items()
    ]
    grade_lines = [
        f"- {grade}: {description}"
        for grade, description in KNOWLEDGE_BASE["oil_grades"].items()
    ]

    return "\n".join(
        [
            KNOWLEDGE_BASE["assistant_scope"],
            "",
            "Oil types:",
            *oil_lines,
            "",
            "Oil grades:",
            *grade_lines,
            "",
            "Export markets:",
            *market_lines,
            "",
            f"Demand index: {KNOWLEDGE_BASE['demand_index']}",
            f"Recommended price: {KNOWLEDGE_BASE['recommended_price']}",
            f"Festival season: {KNOWLEDGE_BASE['festival_season']}",
            f"User benefits: {KNOWLEDGE_BASE['benefits']}",
            f"Competitors: {KNOWLEDGE_BASE['competitors']}",
        ]
    )
