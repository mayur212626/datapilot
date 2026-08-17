"""Runs the model-written pandas snippets against the uploaded dataframe.

Executing generated code is the risky part of any data agent, so we keep a
tight lid on it: a restricted set of builtins, no file or network access, and
only the dataframe plus pandas/numpy/plt in scope. It's not a hardened jail —
for a local analyst tool it's the sensible middle ground.
"""

import io
import contextlib

import matplotlib

matplotlib.use("Agg")  # headless — we never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# builtins we allow the snippet to touch
_SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "round": round, "abs": abs, "enumerate": enumerate,
    "zip": zip, "list": list, "dict": dict, "set": set, "float": float,
    "int": int, "str": str, "print": print,
}


def run_code(code, df, chart_path=None):
    """Execute a snippet with `df` available. Capture stdout as the result.

    If the snippet draws a chart and chart_path is given, we save whatever is
    on the current figure. Returns (stdout_text, made_chart_bool).
    """
    scope = {
        "__builtins__": _SAFE_BUILTINS,
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
    }

    buffer = io.StringIO()
    made_chart = False

    plt.close("all")
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, scope)  # noqa: S102 - intentional, sandboxed above

        if chart_path and plt.get_fignums():
            plt.savefig(chart_path, bbox_inches="tight", dpi=120)
            made_chart = True
    except Exception as e:
        return f"Error while running analysis: {e}", False
    finally:
        plt.close("all")

    out = buffer.getvalue().strip()
    return (out or "(no printed output)"), made_chart
