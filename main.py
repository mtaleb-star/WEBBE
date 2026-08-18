import os

import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "Diabetes_Model.pkl")

clf = joblib.load(MODEL_PATH)

FEATURES = ["Glucose", "BMI", "Age", "BloodPressure"]

# min, max, human label for validation + error messages
RANGES = {
    "Glucose": (50, 200, "glucose"),
    "BMI": (15, 60, "bmi"),
    "Age": (21, 81, "age"),
    "BloodPressure": (40, 122, "blood pressure"),
}

# Every accepted spelling normalizes (lowercase, underscores stripped) to
# one of these keys, which then maps to the canonical feature name.
ALIASES = {
    "glucose": "Glucose",
    "bmi": "BMI",
    "age": "Age",
    "bloodpressure": "BloodPressure",
    "bp": "BloodPressure",
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_key(key: str) -> str:
    return key.strip().lower().replace("_", "")


def extract_features(body: dict) -> dict:
    normalized = {}
    for raw_key, value in body.items():
        norm = normalize_key(str(raw_key))
        canonical = ALIASES.get(norm)
        if canonical:
            normalized[canonical] = value

    values = {}
    for feature in FEATURES:
        lo, hi, label = RANGES[feature]

        if feature not in normalized:
            raise HTTPException(
                status_code=400,
                detail=f"Missing value for {label}. Expected a number between {lo} and {hi}.",
            )

        raw = normalized[feature]
        try:
            num = float(raw)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail=f"'{label}' must be a number, got {raw!r}.",
            )

        if not (lo <= num <= hi):
            raise HTTPException(
                status_code=400,
                detail=f"'{label}' must be between {lo} and {hi}, got {num}.",
            )

        values[feature] = num

    return values


def run_screen(values: dict) -> dict:
    row = [[values[f] for f in FEATURES]]
    proba = clf.predict_proba(row)[0]
    classes = list(clf.classes_)

    # class 1 = HIGH risk, class 0 = LOW risk
    p_high = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
    p_low = float(proba[classes.index(0)]) if 0 in classes else float(proba[0])

    verdict = "HIGH" if p_high >= p_low else "LOW"
    confidence = max(p_high, p_low)

    drivers = []
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        total = float(sum(importances)) or 1.0
        raw_drivers = [
            {
                "name": FEATURES[i],
                "importance": float(importances[i]),
                "percent": round(float(importances[i]) / total * 100, 1),
            }
            for i in range(len(FEATURES))
        ]
        drivers = sorted(raw_drivers, key=lambda d: d["importance"], reverse=True)

    return {
        "verdict": verdict,
        "risk": verdict,
        "label": verdict,
        "confidence": confidence,
        "probability": p_high,
        "probability_high": p_high,
        "probability_low": p_low,
        "drivers": drivers,
        "inputs": values,
    }


async def handle_request(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be JSON.")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    values = extract_features(body)
    return run_screen(values)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    return await handle_request(request)


@app.post("/screen")
async def screen(request: Request):
    return await handle_request(request)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
