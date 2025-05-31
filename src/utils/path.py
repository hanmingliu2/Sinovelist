from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FOLDER = PROJECT_ROOT / "src" / "config"
UTILS_FOLDER = PROJECT_ROOT / "src" / "utils"
DATA_FOLDER = PROJECT_ROOT / "src" / "data"
MODELS_FOLDER = PROJECT_ROOT / "src" / "models"
NOVELS_FOLDER = DATA_FOLDER / "novels"
NOVELS_METADATA = DATA_FOLDER / "novel_metadata.csv"
