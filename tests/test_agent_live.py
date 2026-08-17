"""End-to-end smoke test of the full agent loop against the sample data.

This one actually calls the model, so it needs OPENAI_API_KEY in your
environment (or .env). It's kept separate from the offline checks so CI and
quick local runs don't depend on network or spend.

    python tests/test_agent_live.py
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agent.analyst import analyze

load_dotenv()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping live test: OPENAI_API_KEY not set.")
        return

    df = pd.read_csv("sample_data/sales.csv")
    chart = Path("charts")
    chart.mkdir(exist_ok=True)
    chart_path = chart / "test.png"

    question = "Which region is losing revenue over the year, and roughly how much did it drop?"
    print(f"Q: {question}\n")

    answer, made_chart = analyze(df, question, str(chart_path))

    print("A:", answer)
    print("\nChart produced:", made_chart)

    # sanity: the sample data was built so West declines — a working agent
    # should surface that. Not a hard assert (model wording varies), just a hint.
    if "west" in answer.lower():
        print("\nLooks right — agent identified the declining region.")
    else:
        print("\nNote: expected 'West' to appear. Review the answer above.")


if __name__ == "__main__":
    main()
