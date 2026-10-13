API_KEY = ""
API_BASE = ""

QWEN_API_KEY = ""
QWEN_API_BASE = ""

DEEPSEEK_API_KEY = ""
DEEPSEEK_API_BASE = ""

DEFAULT_TEMPERATURE = 0.7
STREAMING = True

MODELS = [
    {
        "name": "deepseek-v4-pro",
        "api_key": DEEPSEEK_API_KEY,
        "api_base": DEEPSEEK_API_BASE,
    },
    {
        "name": "deepseek-v4-flash",
        "api_key": DEEPSEEK_API_KEY,
        "api_base": DEEPSEEK_API_BASE,
    },
    {
        "name": "claude-sonnet-5",
        "api_key": API_KEY,
        "api_base": API_BASE,
    },
    {
        "name": "qwen3.7-plus",
        "api_key": API_KEY,
        "api_base": API_BASE,
    },
    {
        "name": "claude-sonnet-4-6",
        "api_key": API_KEY,
        "api_base": API_BASE,
    },
    {
        "name": "gpt-5.5",
        "api_key": API_KEY,
        "api_base": API_BASE,
    },
    {
        "name": "gpt-5.6-terra",
        "api_key": API_KEY,
        "api_base": API_BASE,
    },
    {
        "name": "gpt-5.6-luna",
        "api_key": API_KEY,
        "api_base": API_BASE,
    },
]

CATEGORIES = [
    "crypto",
    "web",
    "forensics",
    "misc",
    "reverse",
    "pwn",
]

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT.parent
CYBENCH_DIR = DATA_ROOT / "EvoProbeBench" / "cybench"
FEEDBACK_DIR = DATA_ROOT / "EvoProbeBench" / "feedback"
QUESTION_DIR = DATA_ROOT / "question"

MAX_RETRIES = 5
FILE_SIZE_THRESHOLD = 100 * 1024 * 1024  # 100MB
