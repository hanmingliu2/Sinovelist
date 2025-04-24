from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FOLDER = PROJECT_ROOT / "src" / "config"
UTILS_FOLDER = PROJECT_ROOT / "src" / "utils"
DATA_FOLDER = PROJECT_ROOT / "src" / "data"
NOVELS_FOLDER = DATA_FOLDER / "novels"
METADATA_CSV = DATA_FOLDER / "metadata.csv"
