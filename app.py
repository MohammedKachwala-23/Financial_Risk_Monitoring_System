from flask import Flask, render_template, request, Response
import pandas as pd
import io

from risk_engine import run_risk_analysis

app = Flask(__name__)

RESULT = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    global RESULT

    try:
        file = request.files.get('file')

        if not file:
            return render_template('index.html', error="Upload file")

        # READ FILE
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # FORCE CLEAN STRUCTURE
        df = df.copy()
        for col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        # RUN ENGINE
        df = run_risk_analysis(df)
        RESULT = df

        # SUMMARY
        summary = df["Risk_Level"].value_counts().to_dict()
        total = len(df)

        # ─────────────────────────────────────────
        # HOURLY CHART (SAFE)
        # ─────────────────────────────────────────
        hourly = pd.DataFrame(0, index=range(24), columns=["Low", "Medium", "High"])

        try:
            temp = df.groupby("Hour")["Risk_Level"].value_counts().unstack(fill_value=0)
            for col in temp.columns:
                hourly[col] = temp[col]
        except:
            pass

        hour_labels = list(range(24))
        hour_high = hourly["High"].tolist()
        hour_medium = hourly["Medium"].tolist()

        # ─────────────────────────────────────────
        # ✅ FIXED: TOP RISK DEPARTMENTS
        # ─────────────────────────────────────────
        high_df = df[df["Risk_Level"] == "High"]

        # Fallback if no high risk
        if high_df.empty:
            high_df = df.sort_values("Hybrid_Score", ascending=False).head(20)

        dept = (
            high_df.groupby("Department")
            .size()
            .sort_values(ascending=False)
            .head(8)
        )

        dept_labels = dept.index.tolist()
        dept_values = dept.values.tolist()

        # ─────────────────────────────────────────
        # TABLE
        # ─────────────────────────────────────────
        table_df = df.sort_values("Hybrid_Score", ascending=False).head(20)
        table_html = table_df.to_html(index=False)

        return render_template(
            "dashboard.html",
            summary=summary,
            total=total,
            hour_labels=hour_labels,
            hour_high=hour_high,
            hour_medium=hour_medium,
            dept_labels=dept_labels,
            dept_values=dept_values,
            table=table_html
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


@app.route('/export')
def export():
    global RESULT

    if RESULT is None:
        return "Run analysis first"

    output = io.StringIO()
    RESULT.to_csv(output, index=False)

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment; filename=result.csv"}
    )


if __name__ == "__main__":
    app.run(debug=True)