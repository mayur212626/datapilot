# DataPilot

An AI analyst for your spreadsheets. Upload a CSV, ask a question in plain
English, and it investigates the data step by step — writing and running real
pandas — then answers with a chart and a concrete recommendation.

Built to replace the hour a junior analyst spends poking at a file just to
answer one question.

## Why it's different

This isn't a chatbot that *talks about* your data. It's an agent that *works*
your data: it profiles the file, writes small pandas snippets, runs them,
reads the output, and decides whether it has enough to answer or needs another
step. Every number in the answer comes from code that actually ran — not from
the model guessing.

## Stack

- **Agent loop:** OpenAI tool-calling
- **Analysis:** pandas + numpy, run in a restricted sandbox
- **Charts:** matplotlib (headless)
- **UI:** Streamlit

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then add your OpenAI key
```

## Run

```bash
streamlit run app.py
```

Upload a CSV, type a question, watch it work.

## How it works

1. On upload, the file is auto-profiled (shape, types, missing values, ranges)
   so the agent starts with context.
2. The agent answers by calling a `run_analysis` tool — short pandas snippets
   that run against the dataframe.
3. Snippets execute in a restricted sandbox: limited builtins, no file or
   network access, only the dataframe and pandas/numpy/matplotlib in scope.
4. Tool output is fed back to the agent, which loops until it can answer — then
   returns a plain-English summary, a chart, and a recommendation.

## Safety note

The agent runs model-generated code. The sandbox blocks file and network
access and restricts builtins, which is appropriate for a local analyst tool.
For a multi-tenant or hosted deployment, move execution into a proper isolated
container.

## Roadmap

- Support Excel and multiple sheets
- Export the analysis as a shareable report
- Remember prior questions within a session for follow-ups
- Anomaly detection and forecasting on time-series columns
