"""
NutriForge AI — Indian Diet Planner API
Production-ready. Deploy via: uvicorn nutriforge_api:app --host 0.0.0.0 --port $PORT
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutriforge")

CSV_PATH = Path(__file__).parent / "indian_diet_data.csv"
if not CSV_PATH.exists():
    CSV_PATH = Path("indian_diet_data.csv")

df = None
CSV_LOAD_ERROR = None
_food_cache: dict = {}

try:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    for col in ["meal_type", "tags", "goal_tags", "common_allergies"]:
        df[col] = df[col].fillna("").apply(
            lambda x: [i.strip() for i in str(x).split(";") if i.strip()]
        )
    df["_eff"] = df["protein_g"] / df["calories"].clip(lower=1)
    logger.info(f"[NutriForge] Loaded {len(df)} Indian food items.")
except Exception as e:
    CSV_LOAD_ERROR = str(e)
    logger.error(f"[NutriForge] CSV load failed: {e}")


def require_data():
    if df is None:
        raise HTTPException(
            status_code=503,
            detail=f"Dataset unavailable: {CSV_LOAD_ERROR}. Ensure 'indian_diet_data.csv' is deployed alongside the app."
        )


TRAINING_DATA = [
    (22, 55, 162, "female", "fat_loss",    1350),
    (25, 70, 175, "male",   "muscle_gain", 2800),
    (30, 80, 180, "male",   "maintenance", 2400),
    (28, 60, 165, "female", "maintenance", 1800),
    (35, 90, 178, "male",   "fat_loss",    2000),
    (45, 75, 170, "male",   "fat_loss",    1900),
    (22, 50, 158, "female", "muscle_gain", 2100),
    (40, 85, 182, "male",   "muscle_gain", 3100),
    (32, 65, 168, "female", "fat_loss",    1500),
    (27, 73, 177, "male",   "maintenance", 2350),
    (50, 95, 175, "male",   "fat_loss",    2100),
    (24, 48, 155, "female", "fat_loss",    1200),
    (38, 68, 163, "female", "maintenance", 1900),
    (29, 82, 183, "male",   "muscle_gain", 3200),
    (33, 77, 176, "male",   "fat_loss",    2050),
    (21, 57, 160, "female", "maintenance", 1700),
    (44, 100,181, "male",   "fat_loss",    2300),
    (26, 63, 172, "male",   "maintenance", 2200),
    (31, 54, 157, "female", "muscle_gain", 2000),
    (48, 72, 169, "male",   "maintenance", 2100),
    (23, 46, 153, "female", "fat_loss",    1150),
    (36, 88, 179, "male",   "muscle_gain", 3050),
    (42, 78, 174, "male",   "maintenance", 2250),
    (27, 61, 164, "female", "muscle_gain", 2150),
    (55, 82, 171, "male",   "fat_loss",    1950),
    (19, 65, 178, "male",   "muscle_gain", 2900),
    (34, 58, 161, "female", "maintenance", 1750),
    (29, 95, 184, "male",   "muscle_gain", 3300),
    (41, 67, 166, "female", "fat_loss",    1450),
    (25, 78, 181, "male",   "fat_loss",    2100),
    (38, 55, 159, "female", "muscle_gain", 2050),
    (52, 88, 176, "male",   "maintenance", 2200),
    (30, 62, 167, "female", "fat_loss",    1400),
    (20, 72, 176, "male",   "maintenance", 2400),
    (45, 58, 162, "female", "maintenance", 1650),
    (28, 84, 180, "male",   "fat_loss",    2200),
    (37, 52, 156, "female", "fat_loss",    1250),
    (24, 91, 185, "male",   "muscle_gain", 3400),
    (43, 74, 173, "male",   "fat_loss",    2000),
    (31, 60, 163, "female", "muscle_gain", 2100),
]

_gender_enc = LabelEncoder()
_goal_enc   = LabelEncoder()
_scaler     = StandardScaler()
_model      = LinearRegression()


def _train_model():
    data = pd.DataFrame(TRAINING_DATA,
                        columns=["age", "weight", "height", "gender", "goal", "calories"])
    data["gender_enc"] = _gender_enc.fit_transform(data["gender"])
    data["goal_enc"]   = _goal_enc.fit_transform(data["goal"])
    features = data[["age", "weight", "height", "gender_enc", "goal_enc"]].values
    targets  = data["calories"].values
    scaled   = _scaler.fit_transform(features)
    _model.fit(scaled, targets)
    logger.info("[NutriForge] ML calorie prediction model trained successfully.")


_train_model()


def predict_calories(age: int, weight: float, height: float,
                     gender: str, goal: str) -> int:
    try:
        g_enc  = _gender_enc.transform([gender.lower()])[0]
        gl_enc = _goal_enc.transform([goal.lower()])[0]
    except ValueError:
        g_enc  = 0
        gl_enc = 0
    features = np.array([[age, weight, height, g_enc, gl_enc]], dtype=float)
    scaled   = _scaler.transform(features)
    return max(1000, int(round(_model.predict(scaled)[0])))


def bmi_calorie_adjustment(bmi: float, goal: str) -> float:
    if goal == "fat_loss":
        if bmi >= 30:   return 0.92
        if bmi >= 27:   return 0.96
        if bmi < 18.5:  return 1.05
    elif goal == "muscle_gain":
        if bmi < 18.5:  return 1.12
        if bmi >= 30:   return 1.03
    elif goal == "maintenance":
        if bmi < 18.5:  return 1.08
        if bmi >= 30:   return 0.95
    return 1.0


class UserProfile(BaseModel):
    age: int = Field(..., ge=10, le=100)
    height_cm: float = Field(..., ge=100, le=250)
    weight_kg: float = Field(..., ge=20, le=300)
    gender: str = Field(..., description="'male' or 'female'")
    goal: str = Field(..., description="'fat_loss' | 'muscle_gain' | 'maintenance'")
    activity: str = Field(..., description="'sedentary'|'light'|'moderate'|'very_active'|'athlete'")
    diet_type: str = Field("nonveg", description="'veg' or 'nonveg'")
    meals_per_day: int = Field(3, ge=3, le=5)
    allergies: list[str] = Field(default=[], description="e.g. ['dairy','nuts','gluten']")


class ChatRequest(BaseModel):
    message: str
    profile: Optional[UserProfile] = None


class FoodQuery(BaseModel):
    goal: Optional[str] = None
    diet_type: Optional[str] = "nonveg"
    meal_type: Optional[str] = None
    allergies: Optional[list[str]] = []
    limit: Optional[int] = 10


ACTIVITY_MULT = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "very_active": 1.725, "athlete": 1.9}
GOAL_FACTOR   = {"fat_loss": 0.80, "maintenance": 1.00, "muscle_gain": 1.10}
PROTEIN_TGT   = {"fat_loss": (1.8, 2.2), "muscle_gain": (2.2, 2.8), "maintenance": (1.6, 2.0)}
MACRO_RATIOS  = {
    "fat_loss":    {"protein": 0.35, "carbs": 0.40, "fat": 0.25},
    "muscle_gain": {"protein": 0.30, "carbs": 0.50, "fat": 0.20},
    "maintenance": {"protein": 0.25, "carbs": 0.50, "fat": 0.25},
}
CAL_SPLITS = {
    3: {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30},
    4: {"breakfast": 0.25, "mid_morning_snack": 0.10, "lunch": 0.40, "dinner": 0.25},
    5: {"breakfast": 0.25, "mid_morning_snack": 0.10, "lunch": 0.35, "afternoon_snack": 0.10, "dinner": 0.20},
}
SLOT_TYPE = {
    "breakfast": "breakfast", "mid_morning_snack": "snack",
    "lunch": "lunch", "afternoon_snack": "snack", "dinner": "dinner",
}
SLOT_REQUIRED_CATS = {
    "breakfast":         [("carb", True),  ("protein", True),     ("dairy", False), ("fruit", False)],
    "mid_morning_snack": [("fruit", False), ("dairy", False),      ("protein", False)],
    "lunch":             [("protein", True),("carb", True),        ("vegetable", True), ("fat", False)],
    "afternoon_snack":   [("protein", False),("fruit", False),     ("dairy", False)],
    "dinner":            [("protein", True),("vegetable", True),   ("carb", False), ("fat", False)],
}
INDIAN_COMBOS = {
    "lunch":     [("protein", "carb"), ("carb", "vegetable")],
    "dinner":    [("protein", "vegetable"), ("carb", "protein")],
    "breakfast": [("carb", "protein"), ("carb", "dairy")],
}
DINNER_CAL_FACTOR = {"fat_loss": 0.85, "muscle_gain": 1.0, "maintenance": 0.93}


def calc_bmi(w, h): return round(w / (h / 100) ** 2, 1)

def bmi_cat(b):
    if b < 18.5: return "Underweight"
    if b < 25:   return "Normal"
    if b < 30:   return "Overweight"
    return "Obese"

def calc_bmr(w, h, age, gender):
    base = 10 * w + 6.25 * h - 5 * age
    return round(base + 5 if gender.lower() == "male" else base - 161, 1)

def calc_tdee(bmr, activity):
    return round(bmr * ACTIVITY_MULT.get(activity.lower(), 1.55), 1)

def calc_target(tdee, goal):
    return round(tdee * GOAL_FACTOR.get(goal, 1.0))

def protein_range(w, goal):
    lo, hi = PROTEIN_TGT.get(goal, (1.6, 2.0))
    return {"min_g": round(lo * w), "max_g": round(hi * w)}


def _base_filter(slot_type: str, diet_type: str, allergies: list) -> pd.DataFrame:
    key = f"{slot_type}|{diet_type}|{','.join(sorted(allergies))}"
    if key in _food_cache:
        return _food_cache[key]
    mask = df["meal_type"].apply(lambda mt: slot_type in mt)
    if diet_type == "veg":
        mask &= df["diet_type"] == "veg"
    if allergies:
        la = [a.lower() for a in allergies]
        mask &= df["common_allergies"].apply(lambda al: not any(a in al for a in la))
    result = df[mask].copy()
    _food_cache[key] = result
    return result


def filter_foods(slot_type: str, goal: str, diet_type: str, allergies: list) -> pd.DataFrame:
    base = _base_filter(slot_type, diet_type, allergies)
    if goal and not base.empty:
        gm = base["goal_tags"].apply(lambda gt: goal in gt or "all" in gt)
        filtered = base[gm]
        if not filtered.empty:
            return filtered
    return base


def _weighted_pick(grp: pd.DataFrame, goal: str, cat: str) -> pd.Series:
    if grp.empty:
        return grp
    if goal == "muscle_gain" and cat == "protein":
        raw = grp["protein_g"]
    elif goal == "fat_loss" and cat in ("protein", "vegetable"):
        raw = grp["_eff"]
    elif goal == "fat_loss" and cat == "carb":
        raw = 1.0 / grp["calories"].clip(lower=1)
    else:
        raw = grp.get("health_score", pd.Series(1.0, index=grp.index))
    raw = raw.fillna(0.5).clip(lower=0.01)
    w   = raw / raw.sum()
    return grp.sample(1, weights=w).iloc[0]


def pick_foods(slot_name: str, target_cal: int, goal: str,
               diet_type: str, allergies: list, protein_budget_g: float,
               used_global: set) -> list:
    slot_type = SLOT_TYPE.get(slot_name, "lunch")
    slot_reqs = SLOT_REQUIRED_CATS.get(slot_name, [("protein", True), ("carb", True)])

    cands = filter_foods(slot_type, goal, diet_type, allergies)
    if cands.empty:
        cands = _base_filter(slot_type, diet_type, allergies)
    if cands.empty:
        cands = _base_filter(slot_type, "nonveg", [])
    if cands.empty:
        return []

    if slot_name == "dinner":
        target_cal = round(target_cal * DINNER_CAL_FACTOR.get(goal, 0.93))

    selected, used_cal, used_prot = [], 0, 0.0
    used_ids = set(used_global)
    seen_cats: list[str] = []

    for cat, required in slot_reqs:
        if used_cal >= target_cal * 0.92:
            break
        grp = cands[(cands["category"] == cat) & (~cands["food_id"].isin(used_ids))]
        if grp.empty:
            grp = cands[cands["category"] == cat] if required else grp
        if grp.empty:
            continue
        row = _weighted_pick(grp, goal, cat)
        if used_cal + row["calories"] <= target_cal * 1.25:
            selected.append(row)
            used_cal  += row["calories"]
            used_prot += row["protein_g"]
            used_ids.add(row["food_id"])
            seen_cats.append(cat)

    for a, b in INDIAN_COMBOS.get(slot_name, []):
        if a in seen_cats and b not in seen_cats:
            grp = cands[(cands["category"] == b) & (~cands["food_id"].isin(used_ids))]
            if not grp.empty:
                row = _weighted_pick(grp, goal, b)
                if used_cal + row["calories"] <= target_cal * 1.30:
                    selected.append(row)
                    used_cal  += row["calories"]
                    used_prot += row["protein_g"]
                    used_ids.add(row["food_id"])
                    seen_cats.append(b)
                    break

    if used_prot < protein_budget_g * 0.55:
        grp = cands[(cands["category"] == "protein") & (~cands["food_id"].isin(used_ids))]
        if not grp.empty:
            row = _weighted_pick(grp, goal, "protein")
            if used_cal + row["calories"] <= target_cal * 1.35:
                selected.append(row)
                used_cal  += row["calories"]
                used_ids.add(row["food_id"])

    if used_cal < target_cal * 0.58:
        extras = cands[~cands["food_id"].isin(used_ids)]
        if not extras.empty:
            row = _weighted_pick(extras, goal, "carb")
            selected.append(row)
            used_ids.add(row["food_id"])

    used_global.update(r["food_id"] for r in selected)

    return [{
        "name":      r["name"],
        "quantity":  f"{r['quantity']} {r['quantity_unit']}",
        "calories":  int(r["calories"]),
        "protein_g": float(r["protein_g"]),
        "carbs_g":   float(r["carbs_g"]),
        "fat_g":     float(r["fat_g"]),
        "category":  r["category"],
        "diet_type": r["diet_type"],
    } for r in selected]


def generate_meal_plan(p: UserProfile):
    bmi  = calc_bmi(p.weight_kg, p.height_cm)
    bmr  = calc_bmr(p.weight_kg, p.height_cm, p.age, p.gender)
    tdee = calc_tdee(bmr, p.activity)

    ml_calories      = predict_calories(p.age, p.weight_kg, p.height_cm, p.gender, p.goal)
    formula_calories = calc_target(tdee, p.goal)
    bmi_adj          = bmi_calorie_adjustment(bmi, p.goal)
    tgt              = round(((ml_calories + formula_calories) / 2) * bmi_adj)

    pro           = protein_range(p.weight_kg, p.goal)
    per_meal_prot = pro["min_g"] / p.meals_per_day
    used_global   = set()

    meals, tc, tp, tcarb, tf = [], 0, 0, 0, 0

    for slot, frac in CAL_SPLITS[p.meals_per_day].items():
        mt    = round(tgt * frac)
        foods = pick_foods(slot, mt, p.goal, p.diet_type, p.allergies, per_meal_prot, used_global)
        mc    = sum(f["calories"] for f in foods)
        mp    = round(sum(f["protein_g"] for f in foods), 1)
        mcarb = round(sum(f["carbs_g"]   for f in foods), 1)
        mf    = round(sum(f["fat_g"]     for f in foods), 1)
        tc += mc; tp += mp; tcarb += mcarb; tf += mf
        meals.append({
            "meal_name":       slot.replace("_", " ").title(),
            "target_calories": mt,
            "actual_calories": mc,
            "protein_g":       mp,
            "carbs_g":         mcarb,
            "fat_g":           mf,
            "foods":           foods,
        })

    return {
        "stats": {
            "bmi":                   bmi,
            "bmi_category":          bmi_cat(bmi),
            "bmr_kcal":              bmr,
            "tdee_kcal":             tdee,
            "ml_predicted_calories": ml_calories,
            "formula_calories":      formula_calories,
            "bmi_adjustment_factor": round(bmi_adj, 3),
            "target_calories":       tgt,
            "goal":                  p.goal,
            "protein_target":        pro,
            "macro_targets":         MACRO_RATIOS.get(p.goal, MACRO_RATIOS["maintenance"]),
        },
        "meals": meals,
        "daily_totals": {
            "calories":  tc,
            "protein_g": round(tp,    1),
            "carbs_g":   round(tcarb, 1),
            "fat_g":     round(tf,    1),
        },
    }


INTENTS = {
    "bmi":       r"\bbmi\b",
    "bmr":       r"\bbmr\b",
    "tdee":      r"\btdee\b|total daily energy|maintenance calories",
    "meal_plan": r"meal plan|diet plan|what (should|can) i eat|plan my diet|plan for me|generate plan|create plan|make plan",
    "food_list": r"(best|good|list|suggest|recommend|show).*(food|meal|dish|snack|recipe|item|eat)|foods? (for|to)",
    "calories":  r"calori|how (much|many) (should|do) i eat|caloric",
    "protein":   r"protein (intake|target|need|require)|how (much|many) protein",
    "greeting":  r"^(hi|hello|hey|namaste|hii|sup)\b",
    "help":      r"\bhelp\b|what can you do|capabilities|features",
}
GOAL_KW = {
    "fat_loss":    ["lose weight", "fat loss", "weight loss", "cut", "slim", "reduce fat", "lose fat"],
    "muscle_gain": ["gain muscle", "muscle gain", "bulk", "build muscle", "gain weight", "mass"],
    "maintenance": ["maintain", "maintenance", "stay fit", "keep weight"],
}
DIET_KW = {
    "veg":    ["vegetarian", "veg ", "veggie", "plant based", "no meat", "no chicken"],
    "nonveg": ["non veg", "nonveg", "meat", "chicken", "fish", "egg", "omnivore"],
}
ACTIVITY_KW = {
    "sedentary":   ["sedentary", "no exercise", "desk job", "inactive"],
    "light":       ["light", "walk", "1-3 days", "little exercise"],
    "moderate":    ["moderate", "3-5 days", "gym 3", "gym 4", "gym 5"],
    "very_active": ["very active", "6-7 days", "daily gym"],
    "athlete":     ["athlete", "twice a day", "professional", "sport"],
}
ALLERGY_KW = {
    "dairy":  ["no dairy", "lactose", "dairy free", "no milk", "no paneer"],
    "gluten": ["gluten free", "no gluten", "no wheat"],
    "nuts":   ["nut allergy", "no nuts", "no almonds", "no peanut"],
    "egg":    ["no egg", "egg allergy", "eggless"],
    "fish":   ["no fish", "fish allergy"],
    "soy":    ["no soy", "soy allergy"],
}


def extract_numbers(text):
    found, lo = {}, text.lower()
    m = re.search(r"(\d+)\s*(?:years?\s*old|yr|years?\s*of\s*age|age\s*:?\s*)", lo)
    if m: found["age"] = int(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*kgs?\b", lo)
    if m: found["weight_kg"] = float(m.group(1))
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*cm\b", lo)
    if m: found["height_cm"] = float(m.group(1))
    return found


def detect(text, mapping):
    lo = text.lower()
    for k, kws in mapping.items():
        if any(kw in lo for kw in kws): return k
    return None


def detect_allergies(text):
    lo = text.lower()
    return [al for al, kws in ALLERGY_KW.items() if any(k in lo for k in kws)]


def detect_intent(text):
    lo = text.lower()
    for intent, pat in INTENTS.items():
        if re.search(pat, lo): return intent
    return "general"


def fmt_plan(plan):
    s = plan["stats"]
    lines = [
        "📊 Your Stats",
        f"  BMI: {s['bmi']} ({s['bmi_category']})",
        f"  BMR: {s['bmr_kcal']} kcal | TDEE: {s['tdee_kcal']} kcal",
        f"  ML Predicted: {s['ml_predicted_calories']} kcal | Formula: {s['formula_calories']} kcal",
        f"  BMI Adjustment: ×{s['bmi_adjustment_factor']}",
        f"  Target (smart): {s['target_calories']} kcal/day ({s['goal'].replace('_',' ').title()})",
        f"  Protein: {s['protein_target']['min_g']}–{s['protein_target']['max_g']} g/day",
        "", "🍽️ Your Indian Meal Plan", "─" * 38,
    ]
    for meal in plan["meals"]:
        lines.append(f"\n{meal['meal_name']} (~{meal['target_calories']} kcal)")
        for f in meal["foods"]:
            lines.append(f"  • {f['name']} — {f['quantity']} → {f['calories']} kcal | P:{f['protein_g']}g C:{f['carbs_g']}g F:{f['fat_g']}g")
        lines.append(f"  Total: {meal['actual_calories']} kcal | P:{meal['protein_g']}g C:{meal['carbs_g']}g F:{meal['fat_g']}g")
    d = plan["daily_totals"]
    lines.append(f"\n📈 Daily: {d['calories']} kcal | P:{d['protein_g']}g | C:{d['carbs_g']}g | F:{d['fat_g']}g")
    return "\n".join(lines)


def chat_logic(message, profile=None):
    intent    = detect_intent(message)
    nums      = extract_numbers(message)
    goal      = detect(message, GOAL_KW)     or (profile.goal      if profile else None)
    diet_type = detect(message, DIET_KW)     or (profile.diet_type if profile else "nonveg")
    activity  = detect(message, ACTIVITY_KW) or (profile.activity  if profile else "moderate")
    allergies = detect_allergies(message)    or (profile.allergies if profile else [])
    age       = nums.get("age")       or (profile.age       if profile else None)
    weight    = nums.get("weight_kg") or (profile.weight_kg if profile else None)
    height    = nums.get("height_cm") or (profile.height_cm if profile else None)
    gender    = profile.gender if profile else "male"

    if intent == "greeting":
        return {"intent": intent, "reply": (
            "Namaste! 🙏 I'm NutriForge AI — your personal Indian diet planner.\n\n"
            "Tell me your age, weight (kg), height (cm), goal (fat loss / muscle gain / maintenance), "
            "and diet (veg/nonveg). I'll build your personalised Indian meal plan!\n\n"
            "Example: 'I am 25 years old, 70 kg, 175 cm, want fat loss, vegetarian'"
        )}

    if intent == "help":
        return {"intent": intent, "reply": (
            "I can help you with:\n"
            "  • 🥗 Personalised Indian meal plans (Roti, Dal, Paneer, Chicken…)\n"
            "  • 📊 BMI, BMR, TDEE calculations\n"
            "  • 🤖 ML-predicted + BMI-adjusted calorie targets\n"
            "  • 🎯 Goals: Fat Loss | Muscle Gain | Maintenance\n"
            "  • 🌿 Veg & Non-Veg plans with allergy filtering\n"
            "  • 💪 Protein & macro targets by goal\n\nJust share your details and I'll get started!"
        )}

    if intent in ("bmi", "bmr", "tdee", "calories", "protein") and weight and height:
        bmr     = calc_bmr(weight, height, age or 25, gender)
        tdee    = calc_tdee(bmr, activity)
        bmi     = calc_bmi(weight, height)
        ml_cal  = predict_calories(age or 25, weight, height, gender, goal or "maintenance")
        formula = calc_target(tdee, goal or "maintenance")
        bmi_adj = bmi_calorie_adjustment(bmi, goal or "maintenance")
        tgt     = round(((ml_cal + formula) / 2) * bmi_adj)
        pro     = protein_range(weight, goal or "maintenance")
        return {"intent": intent, "reply": (
            f"Here are your numbers:\n\n"
            f"  📐 BMI: {bmi} ({bmi_cat(bmi)})\n"
            f"  🔥 BMR: {bmr} kcal/day\n"
            f"  ⚡ TDEE: {tdee} kcal/day\n"
            f"  🤖 ML Predicted: {ml_cal} kcal/day\n"
            f"  📊 BMI Adjustment: ×{round(bmi_adj, 3)}\n"
            f"  🎯 Target ({(goal or 'maintenance').replace('_',' ')}): {tgt} kcal/day\n"
            f"  💪 Protein: {pro['min_g']}–{pro['max_g']} g/day\n\n"
            "Want me to build a full Indian meal plan?"
        ), "data": {"bmi": bmi, "bmi_category": bmi_cat(bmi), "bmr": bmr,
                    "tdee": tdee, "ml_predicted_calories": ml_cal,
                    "bmi_adjustment_factor": round(bmi_adj, 3),
                    "target_calories": tgt, "protein_target_g": pro}}

    if (intent == "meal_plan" or intent == "general") and weight and height:
        p = UserProfile(age=age or 25, height_cm=height, weight_kg=weight, gender=gender,
                        goal=goal or "maintenance", activity=activity or "moderate",
                        diet_type=diet_type or "nonveg",
                        meals_per_day=profile.meals_per_day if profile else 3,
                        allergies=allergies)
        plan = generate_meal_plan(p)
        return {"intent": "meal_plan", "reply": fmt_plan(plan), "data": plan}

    if intent in ("meal_plan", "general"):
        return {"intent": intent, "reply": (
            "To build your meal plan I need:\n"
            "  • Age (years), Weight (kg), Height (cm)\n"
            "  • Goal: fat_loss / muscle_gain / maintenance\n"
            "  • Diet: veg or nonveg\n\n"
            "Example: 'I am 28, 80 kg, 178 cm, muscle gain, non veg'"
        )}

    if intent == "food_list":
        slot  = next((s for s in ["breakfast", "lunch", "dinner", "snack"] if s in message.lower()), "lunch")
        foods = filter_foods(slot, goal or "maintenance", diet_type or "nonveg", allergies).head(10)
        if foods.empty:
            return {"intent": intent, "reply": "No matching foods found. Try adjusting filters."}
        lines = [f"Best Indian foods for {slot}:\n"]
        for _, row in foods.iterrows():
            lines.append(f"  • {row['name']} ({row['quantity']} {row['quantity_unit']}) → {int(row['calories'])} kcal | P:{row['protein_g']}g C:{row['carbs_g']}g F:{row['fat_g']}g")
        return {"intent": intent, "reply": "\n".join(lines)}

    return {"intent": "general", "reply": (
        "I'm NutriForge AI 🍛 — your Indian diet planner.\n\n"
        "Share your age, weight, height, goal and veg/nonveg preference, and I'll create "
        "a personalised Indian meal plan with Dal, Roti, Paneer, Chicken, Rajma and more!\n\n"
        "Or ask me to calculate your BMI, BMR, or daily calories."
    )}


app = FastAPI(
    title="NutriForge Indian Diet AI",
    description="Smart AI-powered Indian diet planner with weighted food selection, BMI-aware calorie adjustment, variety control, and Indian meal intelligence.\n\n"
                "**Features**: BMI/BMR/TDEE, ML + BMI-adjusted calories, balanced macro meals, no food repetition, Indian combos.\n\n"
                "**Dataset**: 80+ common Indian foods with accurate macros.",
    version="3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/ping", tags=["health"])
def ping():
    return PlainTextResponse("OK")


@app.get("/health", tags=["health"])
def health():
    return {
        "status":           "running",
        "dataset_loaded":   df is not None,
        "foods":            len(df) if df is not None else 0,
        "ml_model":         "LinearRegression (trained)",
        "training_samples": len(TRAINING_DATA),
        "version":          "3.0.0",
    }


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "NutriForge Indian Diet AI", "version": "3.0.0",
            "endpoints": ["/chat", "/meal-plan", "/foods", "/calculate", "/health", "/docs"]}


@app.post("/chat", tags=["AI Chat"], summary="Natural language diet assistant")
def chat(req: ChatRequest):
    """
    Send a natural language message. Returns human-readable reply + structured data.

    **Examples:**
    - "I am 25, 70 kg, 175 cm, want fat loss, vegetarian"
    - "What is my BMI if I weigh 80 kg and am 180 cm tall?"
    - "Show me high protein veg breakfast foods"
    """
    try:
        require_data()
        return JSONResponse(content=chat_logic(req.message, req.profile))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/meal-plan", tags=["Meal Plan"], summary="Generate full Indian meal plan")
def meal_plan(profile: UserProfile):
    """Generate a complete day's Indian meal plan with smart calorie adjustment and balanced macros."""
    try:
        require_data()
        return JSONResponse(content=generate_meal_plan(profile))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/meal-plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calculate", tags=["Calculations"], summary="BMI, BMR, TDEE + smart calorie calculator")
def calculate(weight_kg: float, height_cm: float, age: int,
              gender: str = "male", activity: str = "moderate", goal: str = "maintenance"):
    """Returns BMI, BMR, TDEE, ML-predicted calories, BMI-adjusted target, and protein target."""
    try:
        bmi     = calc_bmi(weight_kg, height_cm)
        bmr     = calc_bmr(weight_kg, height_cm, age, gender)
        tdee    = calc_tdee(bmr, activity)
        ml_cal  = predict_calories(age, weight_kg, height_cm, gender, goal)
        formula = calc_target(tdee, goal)
        bmi_adj = bmi_calorie_adjustment(bmi, goal)
        tgt     = round(((ml_cal + formula) / 2) * bmi_adj)
        pro     = protein_range(weight_kg, goal)
        return {"bmi": bmi, "bmi_category": bmi_cat(bmi), "bmr_kcal": bmr,
                "tdee_kcal": tdee, "ml_predicted_calories": ml_cal,
                "formula_calories": formula, "bmi_adjustment_factor": round(bmi_adj, 3),
                "target_calories": tgt, "goal": goal, "protein_target_g": pro,
                "formulas": {"bmi": "weight(kg)/height(m)²", "bmr": "Mifflin-St Jeor",
                             "tdee": f"BMR × {ACTIVITY_MULT.get(activity, 1.55)} ({activity})",
                             "target": "blend(ML + formula) × BMI adjustment"}}
    except Exception as e:
        logger.error(f"/calculate error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/foods", tags=["Foods"], summary="Search Indian foods by filters")
def get_foods(query: FoodQuery):
    """Filter foods by goal, diet type, meal type, and allergies."""
    try:
        require_data()
        results   = filter_foods(query.meal_type or "lunch", query.goal,
                                 query.diet_type or "nonveg", query.allergies or []).head(query.limit or 10)
        drop_cols = [c for c in ["meal_type", "tags", "goal_tags", "common_allergies", "_eff"]
                     if c in results.columns]
        return {"count": len(results), "filters": query.dict(),
                "foods": results.drop(columns=drop_cols).to_dict("records")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/foods error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/foods/all", tags=["Foods"], summary="All foods in database")
def all_foods():
    try:
        require_data()
        drop_cols = [c for c in ["meal_type", "tags", "goal_tags", "common_allergies", "_eff"]
                     if c in df.columns]
        return {"count": len(df), "foods": df.drop(columns=drop_cols).to_dict("records")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/foods/all error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
