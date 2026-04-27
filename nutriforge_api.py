"""
NutriForge AI — Indian Diet Planner API
Production-ready. Deploy via: uvicorn nutriforge_api:app --host 0.0.0.0 --port $PORT
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutriforge")

# ── Load CSV ──────────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "indian_diet_data.csv"
if not CSV_PATH.exists():
    CSV_PATH = Path("indian_diet_data.csv")

df = None
CSV_LOAD_ERROR = None

try:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    for col in ["meal_type", "tags", "goal_tags", "common_allergies"]:
        df[col] = df[col].fillna("").apply(
            lambda x: [i.strip() for i in str(x).split(";") if i.strip()]
        )
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

# ── Schemas ───────────────────────────────────────────────────────────────────

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

# ── Constants ─────────────────────────────────────────────────────────────────

ACTIVITY_MULT = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "very_active": 1.725, "athlete": 1.9}
GOAL_FACTOR   = {"fat_loss": 0.80, "maintenance": 1.00, "muscle_gain": 1.10}
PROTEIN_TGT   = {"fat_loss": (1.8, 2.2), "muscle_gain": (2.2, 2.8), "maintenance": (1.6, 2.0)}
CAL_SPLITS = {
    3: {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30},
    4: {"breakfast": 0.25, "mid_morning_snack": 0.10, "lunch": 0.40, "dinner": 0.25},
    5: {"breakfast": 0.25, "mid_morning_snack": 0.10, "lunch": 0.35, "afternoon_snack": 0.10, "dinner": 0.20},
}
SLOT_TYPE = {"breakfast": "breakfast", "mid_morning_snack": "snack",
             "lunch": "lunch", "afternoon_snack": "snack", "dinner": "dinner"}

# ── Calculations ──────────────────────────────────────────────────────────────

def calc_bmi(w, h): return round(w / (h / 100) ** 2, 1)

def bmi_cat(b):
    if b < 18.5: return "Underweight"
    if b < 25:   return "Normal"
    if b < 30:   return "Overweight"
    return "Obese"

def calc_bmr(w, h, age, gender):
    base = 10*w + 6.25*h - 5*age
    return round(base + 5 if gender.lower() == "male" else base - 161, 1)

def calc_tdee(bmr, activity):
    return round(bmr * ACTIVITY_MULT.get(activity.lower(), 1.55), 1)

def calc_target(tdee, goal):
    return round(tdee * GOAL_FACTOR.get(goal, 1.0))

def protein_range(w, goal):
    lo, hi = PROTEIN_TGT.get(goal, (1.6, 2.0))
    return {"min_g": round(lo * w), "max_g": round(hi * w)}

# ── Food Selection ────────────────────────────────────────────────────────────

def filter_foods(slot_type, goal, diet_type, allergies):
    mask = df["meal_type"].apply(lambda mt: slot_type in mt)
    if diet_type == "veg":
        mask &= df["diet_type"] == "veg"
    if goal:
        mask &= df["goal_tags"].apply(lambda gt: goal in gt or "all" in gt)
    if allergies:
        la = [a.lower() for a in allergies]
        mask &= df["common_allergies"].apply(lambda al: not any(a in al for a in la))
    return df[mask].copy()


def pick_foods(slot_name, target_cal, goal, diet_type, allergies):
    slot_type = SLOT_TYPE.get(slot_name, "lunch")
    cands = filter_foods(slot_type, goal, diet_type, allergies)
    if cands.empty:
        cands = filter_foods(slot_type, None, diet_type, allergies)
    if cands.empty:
        return []

    cat_order = ["protein", "carb", "vegetable", "dairy", "fat", "fruit", "drink"]
    selected, used = [], 0
    for cat in cat_order:
        grp = cands[cands["category"] == cat]
        if grp.empty: continue
        row = grp.sample(1).iloc[0]
        if used + row["calories"] <= target_cal * 1.15:
            selected.append(row)
            used += row["calories"]
        if used >= target_cal * 0.85: break

    if used < target_cal * 0.70:
        ids = [r["food_id"] for r in selected]
        extras = cands[~cands["food_id"].isin(ids)]
        if not extras.empty:
            row = extras.sample(1).iloc[0]
            selected.append(row)

    return [{"name": r["name"], "quantity": f"{r['quantity']} {r['quantity_unit']}",
             "calories": int(r["calories"]), "protein_g": float(r["protein_g"]),
             "carbs_g": float(r["carbs_g"]), "fat_g": float(r["fat_g"]),
             "category": r["category"], "diet_type": r["diet_type"]} for r in selected]

# ── Meal Plan ─────────────────────────────────────────────────────────────────

def generate_meal_plan(p: UserProfile):
    bmi  = calc_bmi(p.weight_kg, p.height_cm)
    bmr  = calc_bmr(p.weight_kg, p.height_cm, p.age, p.gender)
    tdee = calc_tdee(bmr, p.activity)
    tgt  = calc_target(tdee, p.goal)
    pro  = protein_range(p.weight_kg, p.goal)
    meals, tc, tp, tcarb, tf = [], 0, 0, 0, 0

    for slot, frac in CAL_SPLITS[p.meals_per_day].items():
        mt    = round(tgt * frac)
        foods = pick_foods(slot, mt, p.goal, p.diet_type, p.allergies)
        mc    = sum(f["calories"] for f in foods)
        mp    = round(sum(f["protein_g"] for f in foods), 1)
        mcarb = round(sum(f["carbs_g"] for f in foods), 1)
        mf    = round(sum(f["fat_g"] for f in foods), 1)
        tc += mc; tp += mp; tcarb += mcarb; tf += mf
        meals.append({"meal_name": slot.replace("_", " ").title(),
                       "target_calories": mt, "actual_calories": mc,
                       "protein_g": mp, "carbs_g": mcarb, "fat_g": mf, "foods": foods})

    return {"stats": {"bmi": bmi, "bmi_category": bmi_cat(bmi), "bmr_kcal": bmr,
                      "tdee_kcal": tdee, "target_calories": tgt, "goal": p.goal,
                      "protein_target": pro},
            "meals": meals,
            "daily_totals": {"calories": tc, "protein_g": round(tp, 1),
                             "carbs_g": round(tcarb, 1), "fat_g": round(tf, 1)}}

# ── NLP Chat Engine ───────────────────────────────────────────────────────────

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
        f"  Target: {s['target_calories']} kcal/day ({s['goal'].replace('_',' ').title()})",
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
            "  • 🎯 Goals: Fat Loss | Muscle Gain | Maintenance\n"
            "  • 🌿 Veg & Non-Veg plans with allergy filtering\n"
            "  • 💪 Protein targets by goal\n\nJust share your details and I'll get started!"
        )}

    if intent in ("bmi", "bmr", "tdee", "calories", "protein") and weight and height:
        bmr  = calc_bmr(weight, height, age or 25, gender)
        tdee = calc_tdee(bmr, activity)
        bmi  = calc_bmi(weight, height)
        tgt  = calc_target(tdee, goal or "maintenance")
        pro  = protein_range(weight, goal or "maintenance")
        return {"intent": intent, "reply": (
            f"Here are your numbers:\n\n"
            f"  📐 BMI: {bmi} ({bmi_cat(bmi)})\n"
            f"  🔥 BMR: {bmr} kcal/day\n"
            f"  ⚡ TDEE: {tdee} kcal/day\n"
            f"  🎯 Target ({(goal or 'maintenance').replace('_',' ')}): {tgt} kcal/day\n"
            f"  💪 Protein: {pro['min_g']}–{pro['max_g']} g/day\n\n"
            "Want me to build a full Indian meal plan?"
        ), "data": {"bmi": bmi, "bmi_category": bmi_cat(bmi), "bmr": bmr,
                    "tdee": tdee, "target_calories": tgt, "protein_target_g": pro}}

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
        slot = next((s for s in ["breakfast", "lunch", "dinner", "snack"] if s in message.lower()), "lunch")
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

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NutriForge Indian Diet AI",
    description="Self-contained Indian diet planner — no external LLM required.\n\n"
                "**Features**: BMI/BMR/TDEE, Indian meal plans, veg/nonveg + allergy filtering, NL chat.\n\n"
                "**Dataset**: 80 common Indian foods with accurate macros.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "running",
        "dataset_loaded": df is not None,
        "foods": len(df) if df is not None else 0,
    }


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "NutriForge Indian Diet AI", "version": "1.0.0",
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
    """Generate a complete day's Indian meal plan from your profile."""
    try:
        require_data()
        return JSONResponse(content=generate_meal_plan(profile))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/meal-plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calculate", tags=["Calculations"], summary="BMI, BMR, TDEE calculator")
def calculate(weight_kg: float, height_cm: float, age: int,
              gender: str = "male", activity: str = "moderate", goal: str = "maintenance"):
    """Returns BMI, BMR, TDEE, target calories, and protein target."""
    try:
        bmi  = calc_bmi(weight_kg, height_cm)
        bmr  = calc_bmr(weight_kg, height_cm, age, gender)
        tdee = calc_tdee(bmr, activity)
        tgt  = calc_target(tdee, goal)
        pro  = protein_range(weight_kg, goal)
        return {"bmi": bmi, "bmi_category": bmi_cat(bmi), "bmr_kcal": bmr,
                "tdee_kcal": tdee, "target_calories": tgt, "goal": goal,
                "protein_target_g": pro,
                "formulas": {"bmi": "weight(kg)/height(m)²",
                             "bmr": "Mifflin-St Jeor",
                             "tdee": f"BMR × {ACTIVITY_MULT.get(activity, 1.55)} ({activity})",
                             "target": f"TDEE × {GOAL_FACTOR.get(goal, 1.0)} ({goal})"}}
    except Exception as e:
        logger.error(f"/calculate error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/foods", tags=["Foods"], summary="Search Indian foods by filters")
def get_foods(query: FoodQuery):
    """Filter foods by goal, diet type, meal type, and allergies."""
    try:
        require_data()
        results = filter_foods(query.meal_type or "lunch", query.goal,
                               query.diet_type or "nonveg", query.allergies or []).head(query.limit or 10)
        drop_cols = [c for c in ["meal_type", "tags", "goal_tags", "common_allergies"] if c in results.columns]
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
        drop_cols = [c for c in ["meal_type", "tags", "goal_tags", "common_allergies"] if c in df.columns]
        return {"count": len(df), "foods": df.drop(columns=drop_cols).to_dict("records")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/foods/all error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
