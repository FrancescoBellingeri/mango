# Mango Benchmark

Evaluates Mango against the [MongoDB NL-to-mongosh benchmark](https://huggingface.co/datasets/mongodb-eai/natural-language-to-mongosh) — 766 natural language → MongoDB query pairs across the Atlas sample databases.

## Prerequisites

**1. Atlas cluster with sample data loaded**

Load the Atlas sample datasets into your cluster via the Atlas UI ("Load Sample Dataset").  
Databases used: `sample_airbnb`, `sample_analytics`, `sample_geospatial`, `sample_guides`, `sample_mflix`, `sample_restaurants`, `sample_supplies`, `sample_training`, `sample_weatherdata`.

**2. Environment variables**

```bash
export MONGODB_URI="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority"
export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY / GOOGLE_API_KEY
export MANGO_PROVIDER="anthropic"       # optional, default: anthropic
export MANGO_MODEL="claude-sonnet-4-6"  # optional, uses provider default
```

**3. Python dependencies**

```bash
pip install -e ".[anthropic]"   # or [openai] or [gemini]
pip install tqdm                # optional but recommended for progress bar
```

## Running the benchmark

```bash
# Quick run — 50 questions (default SAMPLE_SIZE)
python -m mango_benchmark.runner

# Full run — all 766 questions
python -m mango_benchmark.runner --sample 0

# Different provider
python -m mango_benchmark.runner --provider openai --model gpt-4o --sample 0

# Disable memory layer (for ablation)
python -m mango_benchmark.runner --no-memory

# All options
python -m mango_benchmark.runner --help
```

Results are saved to `mango_benchmark/results/results_<timestamp>.csv`.  
A debug JSONL file (`results_<timestamp>_debug.jsonl`) is written in real time — one line per question.

## Generating a report

```bash
python -m mango_benchmark.report mango_benchmark/results/results_20250417_120000.csv

# Save to file
python -m mango_benchmark.report mango_benchmark/results/results_20250417_120000.csv --out report.md
```

## Metrics

| Metric | Description |
|--------|-------------|
| **SuccessfulExecution (SE)** | Agent returned data without error or timeout |
| **NonEmptyOutput (NEO)** | Result is non-empty (not `[]`, `{}`, or count=0) |
| **ReasonableOutput (RO)** | Result contains no `null` values or empty strings |
| **CorrectOutputFuzzy (CO)** | Result fuzzy-matches the reference (order-insensitive, numbers within 1%) |
| **XMaNeR** | Average of SE + NEO + RO + CO — **primary metric** |

### Official benchmark scores (for comparison)

| Model | XMaNeR |
|-------|--------|
| Claude 3.7 Sonnet | 0.8671 |
| GPT-4o | 0.8253 |

## Configuration

Edit [`config.py`](config.py) to change defaults:

| Constant | Default | Description |
|----------|---------|-------------|
| `SAMPLE_SIZE` | `50` | Questions to run (`None` = all 766) |
| `MAX_ITERATIONS` | `8` | Agent tool-call iterations per question |
| `TIMEOUT_SECONDS` | `60` | Per-question wall-clock timeout |
| `MEMORY_ENABLED` | `True` | ChromaDB memory persists within each database group |
| `PROVIDER` | `"anthropic"` | LLM provider |

## Output CSV columns

```
question_id, database, collection, natural_language_query,
mango_result, reference_output,
successful_execution, non_empty_output, reasonable_output, correct_output_fuzzy,
xmaner, token_input, token_output, latency_seconds, complexity, error_detail
```

`mango_result` is the JSON string of rows returned by the last `run_mql` tool call, or an error message.

## How it works

1. Benchmark CSV is downloaded once and cached to `data/benchmark.csv`.
2. Questions are grouped by `databaseName`. One `MangoAgent` is built per database (schema introspection runs once per database).
3. Each question gets a **fresh conversation** via `agent.new_session()`, but shares schema and (optionally) ChromaDB memory with other questions in the same database.
4. The last `run_mql` tool result is captured via the `on_tool_call` callback and used for metric computation.
5. Questions that timeout or throw are marked `SE=0` and all other metrics `0`.
