from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# MIND-small zips extract into a nested folder per split
TRAIN_DIR = DATA_DIR / "train" / "MINDsmall_train"
DEV_DIR = DATA_DIR / "dev" / "MINDsmall_dev"

ARTICLE_SCORES_CSV = DATA_DIR / "article_scores.csv"
CLICK_QUALITY_CSV = DATA_DIR / "click_quality.csv"
Y_TILDE_CSV = DATA_DIR / "y_tilde.csv"
EMBEDDINGS_NPZ = DATA_DIR / "article_embeddings.npz"

BASELINE_MODEL = MODELS_DIR / "baseline.pt"
INFORMATIVE_MODEL = MODELS_DIR / "informative.pt"
