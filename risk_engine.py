import pandas as pd
import numpy as np

# Optional ML
try:
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False


CONFIG = {
    "LARGE_TX_MULTIPLIER": 3,
    "ROUND_MIN": 500,
    "VELOCITY_THRESHOLD": 3,
    "ML_WEIGHT": 0.4,
    "RULE_WEIGHT": 0.6
}


# ─────────────────────────────────────────────
# SAFE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()

    # Ensure required columns exist
    required = ["Transaction_Amount", "Department", "Vendor_ID"]
    for col in required:
        if col not in df.columns:
            df[col] = "Unknown"

    # Convert numeric safely
    df["Transaction_Amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce").fillna(0)

    # Handle time
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Hour"] = df["Timestamp"].dt.hour
        df["Day"] = df["Timestamp"].dt.day

    if "Hour" not in df.columns:
        df["Hour"] = 0

    if "Day" not in df.columns:
        df["Day"] = 1

    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").fillna(0).astype(int)
    df["Day"] = pd.to_numeric(df["Day"], errors="coerce").fillna(1).astype(int)

    return df


# ─────────────────────────────────────────────
# ML SCORE
# ─────────────────────────────────────────────
def ml_score(df):
    if not ML_AVAILABLE:
        return pd.Series(0.0, index=df.index)

    df_ml = pd.DataFrame({
        "amt": df["Transaction_Amount"],
        "hour": df["Hour"],
        "day": df["Day"]
    })

    df_ml = df_ml.fillna(0)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df_ml)

    scores = model.decision_function(df_ml)

    min_s, max_s = scores.min(), scores.max()
    if max_s == min_s:
        return pd.Series(0.0, index=df.index)

    return pd.Series(1 - (scores - min_s) / (max_s - min_s), index=df.index)


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────
def run_risk_analysis(df):

    df = preprocess(df)

    df["Risk_Score"] = 0
    df["Risk_Reason"] = ""

    dept_avg = df.groupby("Department")["Transaction_Amount"].transform("mean")

    # RULES
    large_tx = df["Transaction_Amount"] > 3 * dept_avg
    night_tx = df["Hour"] <= 5
    round_tx = (df["Transaction_Amount"] % 100 == 0) & (df["Transaction_Amount"] >= 500)

    # APPLY RULES
    def add_rule(mask, score, text):
        df.loc[mask, "Risk_Score"] += score
        df.loc[mask, "Risk_Reason"] += text + ", "

    add_rule(large_tx, 40, "Large transaction")
    add_rule(night_tx, 15, "Night transaction")
    add_rule(round_tx, 20, "Round amount")

    df["Risk_Reason"] = df["Risk_Reason"].str.rstrip(", ")
    df["Risk_Reason"] = df["Risk_Reason"].replace("", "No anomaly")

    # ML
    mls = ml_score(df)
    df["ML_Score"] = mls

    df["Hybrid_Score"] = (
        df["Risk_Score"] * CONFIG["RULE_WEIGHT"] +
        mls * df["Risk_Score"].max() * CONFIG["ML_WEIGHT"]
    )

    # CLASSIFY
    high = df["Hybrid_Score"].quantile(0.98)
    med = df["Hybrid_Score"].quantile(0.90)

    def label(x):
        if x >= high:
            return "High"
        elif x >= med:
            return "Medium"
        else:
            return "Low"

    df["Risk_Level"] = df["Hybrid_Score"].apply(label)

    return df