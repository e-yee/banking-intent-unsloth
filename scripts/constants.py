from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "sample_data"

TAGS = {
    "info": "\033[34mInfo:\033[0m",
    "success": "\033[32mSuccess:\033[0m",
    "warning": "\033[33mWarning:\033[0m"
}