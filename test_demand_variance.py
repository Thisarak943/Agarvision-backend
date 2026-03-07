from member_modules.oshini_module.demand_module.predictor import DemandPredictor

p = DemandPredictor()

tests = [
    {
        "oil_type": "Silani Ravana",
        "oil_grade": "Premium",
        "market_region": "Middle East",
        "market_country": "UAE",
        "prediction_period": "Next Month",
        "festival_season": False
    },
    {
        "oil_type": "Silani Cobra",
        "oil_grade": "Standard",
        "market_region": "East Asia",
        "market_country": "China",
        "prediction_period": "Next Week",
        "festival_season": False
    },
    {
        "oil_type": "Silani Peacock",
        "oil_grade": "Budget",
        "market_region": "Southeast Asia",
        "market_country": "Singapore",
        "prediction_period": "Next Quarter",
        "festival_season": True
    },
    {
        "oil_type": "Silani Savera",
        "oil_grade": "Premium",
        "market_region": "Europe",
        "market_country": "Spain",
        "prediction_period": "Next Month",
        "festival_season": False
    },
    {
        "oil_type": "Silani Butterfly",
        "oil_grade": "Standard",
        "market_region": "South Asia",
        "market_country": "India",
        "prediction_period": "Next Quarter",
        "festival_season": True
    }
]

print("=" * 60)
print("DEMAND INDEX PREDICTIONS (Updated Model)")
print("=" * 60)

for t in tests:
    result = p.predict(t)
    print(f"{t['oil_type']:20} | {t['market_country']:10} | Demand: {result['demand_index']:3} ({result['demand_level']})")

print("=" * 60)
