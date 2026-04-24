"""Benchmark configuration — edit these constants to change defaults.

All values can also be overridden via CLI flags; see:
    python -m mango_benchmark.runner --help
"""

from __future__ import annotations

# Number of questions to run per model.  None = all 766.
SAMPLE_SIZE: int | None = 50

# Agent settings
MAX_ITERATIONS: int = 8
TIMEOUT_SECONDS: float = 60.0

# Set to False to disable ChromaDB memory (useful for ablation tests).
MEMORY_ENABLED: bool = True

# Path to a JSONL training file to pre-load before the benchmark.
# None = no training data (ablation). Set to compare with/without training.
TRAINING_FILE: str | None = None

# Output directory for result CSV + debug JSONL files.
RESULTS_DIR: str = "mango_benchmark/results"

# Models to benchmark.
# Each entry: provider (str), model (str | None), api_key (str | None).
# When model is None the provider default is used.
# When api_key is None the provider's env variable is used
# (ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY).
MODELS: list[dict] = [
    {"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key": None},
    # {"provider": "openai",    "model": "gpt-4o",           "api_key": None},
    # {"provider": "gemini",    "model": "gemini-2.0-flash",  "api_key": None},
]
