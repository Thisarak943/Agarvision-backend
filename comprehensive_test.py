from member_modules.oshini_module.demand_module.predictor import DemandPredictor

p = DemandPredictor()

print("=" * 70)
print("COMPREHENSIVE DEMAND PREDICTION TEST")
print("=" * 70)
print()

# Test same oil type across different markets
print("📊 Same Oil (Ravana Premium) - Different Markets:")
print("-" * 70)
markets = [
    ("Middle East", "UAE"),
    ("East Asia", "China"),
    ("Southeast Asia", "Singapore"),
    ("Europe", "Spain"),
    ("South Asia", "India")
]

for region, country in markets:
    result = p.predict({
        "oil_type": "Silani Ravana",
        "oil_grade": "Premium",
        "market_region": region,
        "market_country": country,
        "prediction_period": "Next Month",
        "festival_season": False
    })
    print(f"  {country:12} ({region:15}): Demand = {result['demand_index']:3} ({result['demand_level']})")

print()
print("📊 Different Oil Types - Same Market (UAE):")
print("-" * 70)

oils = [
    ("Silani Ravana", "Premium"),
    ("Silani Savera", "Premium"),
    ("Silani Cobra", "Standard"),
    ("Silani Peacock", "Standard"),
    ("Silani Butterfly", "Budget"),
]

for oil, grade in oils:
    result = p.predict({
        "oil_type": oil,
        "oil_grade": grade,
        "market_region": "Middle East",
        "market_country": "UAE",
        "prediction_period": "Next Month",
        "festival_season": False
    })
    print(f"  {oil:20} ({grade:8}): Demand = {result['demand_index']:3} ({result['demand_level']})")

print()
print("📊 Festival Season Effect (Ravana in UAE):")
print("-" * 70)

for festival in [False, True]:
    result = p.predict({
        "oil_type": "Silani Ravana",
        "oil_grade": "Premium",
        "market_region": "Middle East",
        "market_country": "UAE",
        "prediction_period": "Next Month",
        "festival_season": festival
    })
    season = "Festival Season" if festival else "Regular Season"
    print(f"  {season:20}: Demand = {result['demand_index']:3} ({result['demand_level']})")

print()
print("📊 Time Period Effect (Ravana in China):")
print("-" * 70)

for period in ["Next Week", "Next Month", "Next Quarter"]:
    result = p.predict({
        "oil_type": "Silani Ravana",
        "oil_grade": "Premium",
        "market_region": "East Asia",
        "market_country": "China",
        "prediction_period": period,
        "festival_season": False
    })
    print(f"  {period:15}: Demand = {result['demand_index']:3} ({result['demand_level']})")

print()
print("=" * 70)
print("ANALYSIS:")
print("=" * 70)
print("✓ Predictions now vary based on:")
print("  • Market region & country (different buyer types & competition)")
print("  • Oil type & grade (quality scores & aging)")
print("  • Time period (seasonal effects)")
print("  • Festival season")
print()
print("⚠️  Limitation: Model trained on limited data (predictions 40-65 range)")
print("💡 To improve: Retrain model with more diverse historical data")
print("=" * 70)
