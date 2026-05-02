from __future__ import annotations

MODEL_INPUT_COLUMNS = [
    "Oil_Type",
    "Oil_Grade",
    "Market_Region",
    "Market_Country",
    "Prediction_month",
    "Prediction_week",
    "Festival_Season",
]

CURRENT_OIL_PRICES_LKR = {
    "Silani Ravana": 52628,
    "Silani Savera": 44061,
    "Silani Cobra": 14042,
    "Silani Junglefowl": 26314,
    "Silani Butterfly": 26314,
    "Silani Peacock": 26314,
}

OIL_TYPE_ALIASES = {
    "Ravana": "Silani Ravana",
    "Savera": "Silani Savera",
    "Cobra": "Silani Cobra",
    "Junglefowl": "Silani Junglefowl",
    "Butterfly": "Silani Butterfly",
    "Peacock": "Silani Peacock",
}

OIL_GRADES = ["Budget", "Premium", "Standard"]

COUNTRIES_BY_REGION = {
    "Middle East": ["UAE", "Saudi Arabia"],
    "East Asia": ["China", "Taiwan", "Japan", "Hong Kong"],
    "Southeast Asia": ["Malaysia", "Indonesia", "Vietnam", "Singapore"],
    "Europe": ["Spain", "Slovakia"],
    "South Asia": ["India", "Pakistan"],
}

COUNTRY_TO_REGION = {
    country: region
    for region, countries in COUNTRIES_BY_REGION.items()
    for country in countries
}

MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MONTH_NUMBER_TO_NAME = {
    number: name.title()
    for name, number in MONTH_NAME_TO_NUMBER.items()
}

DEMAND_CATEGORY_THRESHOLDS = {
    "low_max": 45.0,
    "medium_max": 65.0,
}

PRICE_RECOMMENDATION_RULES = {
    "Low": {
        "min_discount_percent": 4.0,
        "max_discount_percent": 1.0,
    },
    "Medium": {
        "min_change_percent": -1.5,
        "max_change_percent": 2.5,
    },
    "High": {
        "max_min_premium_percent": 2.0,
        "base_max_premium_percent": 3.0,
        "extra_max_premium_percent": 5.0,
    },
}

KEY_EXPORT_MARKET_NOTES = {
    "Middle East": "Middle East markets are key agarwood oil export destinations, especially UAE and Saudi Arabia.",
    "East Asia": "East Asian markets are important agarwood oil destinations with demand from perfume and luxury markets.",
    "Southeast Asia": "Southeast Asian markets are established agarwood trading destinations.",
    "Europe": "European demand is represented through Spain and Slovakia.",
    "South Asia": "South Asian demand is represented through India and Pakistan.",
}

COUNTRY_REASON_NOTES = {
    "UAE": "UAE is a strong Middle East market for luxury fragrance products.",
    "Saudi Arabia": "Saudi Arabia is a key Middle East destination for agarwood oil exports.",
    "Japan": "Japan shows demand from perfume and luxury product markets.",
    "Hong Kong": "Hong Kong is an important East Asian trading market for agarwood products.",
    "China": "China is a major East Asian market with demand for fragrance and luxury products.",
    "Taiwan": "Taiwan represents a steady East Asian export destination.",
    "Malaysia": "Malaysia is an established Southeast Asian agarwood trading market.",
    "Indonesia": "Indonesia is an established Southeast Asian agarwood market.",
    "Vietnam": "Vietnam is a growing Southeast Asian agarwood market.",
    "Singapore": "Singapore is a regional trading hub for premium fragrance products.",
    "Spain": "Spain represents European demand for agarwood oil exports.",
    "Slovakia": "Slovakia represents European demand in the dataset.",
    "India": "India shows demand from South Asian fragrance and traditional product markets.",
    "Pakistan": "Pakistan represents South Asian demand for agarwood oil.",
}
