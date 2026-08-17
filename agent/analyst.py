"""The agent loop.

Give it a dataframe and a question. It writes pandas code, we run it, feed
the result back, and let it decide whether it has enough to answer or needs
another step. This tool-use loop is what separates a real analyst agent from
a chatbot that just talks about data.
"""

import json
import os

from openai import OpenAI

from .profile import profile_df
from .sandbox import run_code

_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def _get_client():
    """Create the client lazily so the app can load and profile data without
    a key — we only need one once an actual question is asked."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": (
                "Run a short pandas snippet against the dataframe `df`. "
                "Use print() to output anything you want to see. To make a "
                "chart, draw with matplotlib.pyplot as plt (do not call show)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python using df, pd, np, plt.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]


def analyze(df, question, chart_path, max_steps=5):
    """Answer a question about df. Returns (answer_text, made_chart_bool)."""
    system = (
        "You are a careful data analyst. You have a pandas dataframe called "
        "`df`. Investigate by calling run_analysis with small snippets. "
        "Base every claim on real output — never guess numbers. When you have "
        "enough, give a short, plain-English answer a business owner would "
        "understand, ending with one concrete recommendation.\n\n"
        f"Dataframe overview:\n{profile_df(df)}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    made_chart = False
    client = _get_client()

    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message

        # no tool call -> the agent is ready to answer
        if not msg.tool_calls:
            return msg.content, made_chart

        messages.append(msg)

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            output, drew = run_code(args["code"], df, chart_path)
            made_chart = made_chart or drew

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output[:4000],  # keep tool output bounded
                }
            )

    # ran out of steps — return a best effort
    return (
        "I ran several checks but couldn't converge on a clean answer. "
        "Try narrowing the question.",
        made_chart,
    )
