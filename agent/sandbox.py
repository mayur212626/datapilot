"""Runs the model-written pandas snippets against the uploaded dataframe.

Executing generated code is the risky part of any data agent, so we keep a
tight lid on it: a restricted set of builtins, no file or network access, and
only the dataframe plus pandas/numpy/plt in scope. It's not a hardened jail —
for a local analyst tool it's the sensible middle ground.
"""

import ast
import io
import math
import statistics
import contextlib
import importlib

import matplotlib

matplotlib.use("Agg")  # headless — we never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# modules the snippet is allowed to import. Everything the model actually
# needs for analysis, and nothing that touches the filesystem or network.
_ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
    "math", "statistics", "datetime",
}


def _safe_import(name, *args, **kwargs):
    """Restricted __import__: models routinely write `import pandas as pd`
    inside a snippet, so we allow the analysis libraries and block the rest."""
    root = name.split(".")[0]
    if name in _ALLOWED_IMPORTS or root in _ALLOWED_IMPORTS:
        return importlib.import_module(name)
    raise ImportError(f"import of '{name}' is not allowed here")


# builtins we allow the snippet to touch
_SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "round": round, "abs": abs, "enumerate": enumerate,
    "zip": zip, "list": list, "dict": dict, "set": set, "float": float,
    "int": int, "str": str, "print": print, "__import__": _safe_import,
}


def _exec_with_echo(code, scope):
    """Run the snippet, and if the last line is a bare expression, echo its
    value — the way a notebook cell would. Models often write `df.describe()`
    instead of `print(df.describe())`, and without this they'd see nothing
    and never converge.
    """
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        exec(compile(tree, "<snippet>", "exec"), scope)  # noqa: S102
        value = eval(  # noqa: S307 - sandboxed scope
            compile(ast.Expression(last.value), "<snippet>", "eval"), scope
        )
        if value is not None:
            print(value)
    else:
        exec(compile(tree, "<snippet>", "exec"), scope)  # noqa: S102


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
            _exec_with_echo(code, scope)

        if chart_path and plt.get_fignums():
            plt.savefig(chart_path, bbox_inches="tight", dpi=120)
            made_chart = True
    except Exception as e:
        return f"Error while running analysis: {e}", False
    finally:
        plt.close("all")

    out = buffer.getvalue().strip()
    return (out or "(no printed output)"), made_chart
