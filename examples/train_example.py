"""Load trainingset_example.jsonl into ChromaDB memory.

Usage:
    python -m examples.train
    python -m examples.train --file examples/other.jsonl
    python -m examples.train --memory-dir ./my_memory
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from mango.integrations.chromadb import ChromaAgentMemory
from mango.servers.cli.main import _load_training_file

load_dotenv()

DEFAULT_FILE = Path(__file__).parent / "trainingset_example.jsonl"


async def main(file: str, memory_dir: str) -> None:
    memory = ChromaAgentMemory(persist_dir=memory_dir)
    print(f"Memory before: {memory.count()} interactions, {memory.training_count()} training entries")
    await _load_training_file(memory, file)
    print(f"Memory after:  {memory.count()} interactions, {memory.training_count()} training entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk-load training data into Mango memory")
    parser.add_argument("--file", default=str(DEFAULT_FILE), help="JSONL file to import")
    parser.add_argument("--memory-dir", default=".mango_memory", help="ChromaDB directory")
    args = parser.parse_args()
    asyncio.run(main(args.file, args.memory_dir))
