"""Offline checks for the parts that don't need an API key: data profiling
and the sandbox that runs the agent's code. The full agent loop is exercised
separately since it calls the model."""

import pandas as pd

from agent.profile import profile_df
from agent.sandbox import run_code


def _load():
    return pd.read_csv("sample_data/sales.csv")


def test_profile_mentions_columns():
    df = _load()
    summary = profile_df(df)
    for col in ["month", "region", "units_sold", "revenue"]:
        assert col in summary
    assert f"Rows: {len(df)}" in summary
    print("profile OK\n")
    print(summary)


def test_sandbox_runs_and_prints():
    df = _load()
    out, made_chart = run_code(
        "print(df.groupby('region')['revenue'].sum().idxmax())", df
    )
    assert out == "East"  # East has highest total revenue in the sample
    assert made_chart is False
    print("\nsandbox compute OK -> top region:", out)


def test_sandbox_makes_chart(tmp_path):
    df = _load()
    chart = tmp_path / "c.png"
    code = (
        "west = df[df.region=='West']\n"
        "plt.plot(west['month'], west['revenue'])\n"
        "plt.xticks(rotation=45)"
    )
    out, made_chart = run_code(code, df, str(chart))
    assert made_chart is True
    assert chart.exists()
    print("\nsandbox chart OK ->", chart.name, "created")


def test_sandbox_blocks_file_access():
    df = _load()
    out, made_chart = run_code("open('secret.txt','w')", df)
    # open() is not in the allowed builtins, so this must fail gracefully
    assert "Error" in out
    print("\nsandbox safety OK -> file access blocked")


if __name__ == "__main__":
    test_profile_mentions_columns()
    test_sandbox_runs_and_prints()

    import tempfile
    from pathlib import Path

    test_sandbox_makes_chart(Path(tempfile.mkdtemp()))
    test_sandbox_blocks_file_access()
    print("\nAll offline checks passed.")
