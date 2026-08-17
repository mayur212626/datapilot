"""DataPilot — an AI analyst for your spreadsheets.

Upload a CSV, ask a question in plain English, and it investigates the data
step by step — writing and running real pandas — then answers with a chart
and a recommendation. Meant to replace the hour a junior analyst spends
poking at a file to answer one question.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.analyst import analyze
from agent.profile import profile_df

load_dotenv()

st.set_page_config(page_title="DataPilot", page_icon="📊", layout="wide")

st.title("DataPilot")
st.caption("Upload a CSV, ask a question, get an answer with the working shown.")

CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)


uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded is None:
    st.info("Drop a CSV above to begin. Sales, signups, support tickets — anything tabular.")
    st.stop()

df = pd.read_csv(uploaded)

with st.expander("Preview data", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)
    st.text(profile_df(df))


question = st.text_input(
    "Ask a question",
    placeholder="e.g. Which region is losing sales and why?",
)

if question:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("No OPENAI_API_KEY found. Add it to your .env file.")
        st.stop()

    chart_path = CHART_DIR / "latest.png"
    if chart_path.exists():
        chart_path.unlink()

    with st.spinner("Analyzing..."):
        answer, made_chart = analyze(df, question, str(chart_path))

    st.subheader("Answer")
    st.markdown(answer)

    if made_chart and chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
