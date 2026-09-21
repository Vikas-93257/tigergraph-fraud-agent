import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT.parent / "data" / "HHGOA_IEEE")).resolve()
CACHE_DIR = Path(os.getenv("CACHE_DIR", ROOT.parent / "data")).resolve()
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "local")

TG_HOST = os.getenv("TG_HOST", "")
TG_GRAPH = os.getenv("TG_GRAPH", "FraudGraph")
TG_USERNAME = os.getenv("TG_USERNAME", "")
TG_PASSWORD = os.getenv("TG_PASSWORD", "")
TG_SECRET = os.getenv("TG_SECRET", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CASES_DIR = ROOT / "cases"
MONITOR_DIR = ROOT / "monitoring"
MEMORY_FILE = ROOT / "fraud_agent" / "store" / "agent_cases.jsonl"
