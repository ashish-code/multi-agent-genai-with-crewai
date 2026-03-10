"""
Pipeline Safety RAG Crew — entry point.

Run directly:
    uv run run_crew

Or with a custom question:
    uv run python -m pipeline_safety_rag_crew.main \
        --question "What are the cathodic protection requirements for buried pipelines?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load .env if present (development convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set env vars manually in production

from pipeline_safety_rag_crew.crew import PipelineSafetyRAGCrew

# Ensure output directory exists before kickoff
Path("output").mkdir(exist_ok=True)


# ── Default question ──────────────────────────────────────────────────────────

DEFAULT_QUESTION = (
    "What are the pressure testing requirements for steel pipelines "
    "under 49 CFR §192.505?"
)


# ── Run ───────────────────────────────────────────────────────────────────────

def run(question: str = DEFAULT_QUESTION) -> str:
    """Kick off the crew and return the final answer as a string."""
    inputs = {"question": question}

    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70 + "\n")

    result = PipelineSafetyRAGCrew().crew().kickoff(inputs=inputs)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result.raw)
    print("\nAnswer saved to: output/answer.md")

    return result.raw


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline Safety RAG Crew — ask questions about PHMSA regulations"
    )
    parser.add_argument(
        "-q", "--question",
        default=DEFAULT_QUESTION,
        help="Question to answer (default: §192.505 pressure testing requirements)",
    )
    args = parser.parse_args()
    run(args.question)


if __name__ == "__main__":
    main()
