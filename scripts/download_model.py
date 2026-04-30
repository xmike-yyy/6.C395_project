"""
Downloads facebook/bart-large-mnli into models/bart-mnli/ for offline use.
Run this once before running the scoring pipeline.
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).parent.parent / "models" / "bart-mnli"

if __name__ == "__main__":
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f"Model already downloaded at {MODEL_DIR}")
    else:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading facebook/bart-large-mnli (~1.6GB)...")
        AutoTokenizer.from_pretrained("facebook/bart-large-mnli").save_pretrained(MODEL_DIR)
        AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").save_pretrained(MODEL_DIR)
        print(f"Done. Model saved to {MODEL_DIR}")
