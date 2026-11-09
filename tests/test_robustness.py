
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotisense_calibrated.pkl"

model = joblib.load(MODEL_PATH)

test_cases = [
    {
        "category": "Normal",
        "text": "I am extremely happy today",
        "expected": "joy"
    },
    {
        "category": "Normal",
        "text": "I feel completely alone and miserable",
        "expected": "sadness"
    },
    {
        "category": "Negation",
        "text": "I am not happy",
        "expected": "sadness"
    },
    {
        "category": "Negation",
        "text": "I am not afraid anymore",
        "expected": "joy"
    },
    {
        "category": "Spelling Error",
        "text": "I am hapy today",
        "expected": None
    },
    {
        "category": "Spelling Error",
        "text": "I feel terified",
        "expected": None
    },
    {
        "category": "Mixed Emotion",
        "text": "I am happy but nervous about tomorrow",
        "expected": None
    },
    {
        "category": "Mixed Emotion",
        "text": "I love my family but I miss them",
        "expected": None
    },
    {
        "category": "Short Text",
        "text": "Terrible!",
        "expected": "anger"
    },
    {
        "category": "Short Text",
        "text": "Wonderful!",
        "expected": "joy"
    }
]

results = []

for case in test_cases:
    text = case["text"]

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = probabilities.max()

    results.append({
        "Category": case["category"],
        "Text": text,
        "Expected": case["expected"],
        "Predicted": prediction,
        "Probability": confidence
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

results_df.to_csv(
    BASE_DIR / "data" / "processed" / "robustness_results.csv",
    index=False
)

print("\nRobustness results saved successfully.")
