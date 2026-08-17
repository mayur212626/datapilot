"""Quick automatic read of a dataframe so the agent starts with context
instead of going in blind."""


def profile_df(df):
    """Return a compact text summary: shape, columns, types, missing, ranges."""
    lines = [f"Rows: {len(df)}, Columns: {len(df.columns)}", "", "Columns:"]

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        note = f"  - {col} ({dtype})"
        if missing:
            note += f", {missing} missing"

        # give the model a feel for the values
        if df[col].dtype.kind in "biufc":  # numeric
            note += f", range {df[col].min()}–{df[col].max()}"
        else:
            uniques = df[col].dropna().unique()
            if len(uniques) <= 6:
                note += f", values: {list(uniques)}"
            else:
                note += f", {df[col].nunique()} unique"
        lines.append(note)

    return "\n".join(lines)
