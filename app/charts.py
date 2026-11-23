"""
Altair chart helpers for the dashboard.

Colour rules (validated with the dataviz skill's palette checker):
  * Models (3 categories, identity)  -> blue / orange / aqua, fixed order.
  * Risk class (ordered Low<Medium<High) -> ONE-hue blue ramp, light to dark.
  * Magnitude (heat-maps)            -> one-hue blue ramp.
  * Correlation (polarity)           -> blue <-> red diverging, grey midpoint.
Every multi-series chart has a legend and the key numbers are labelled on the
chart or in a table tab, so nothing relies on colour alone.
"""
import altair as alt
import numpy as np
import pandas as pd

from results_data import ERROR_ORDER, MODELS, RISK

MODEL_COLORS = {"Decision Tree": "#2a78d6", "Random Forest": "#eb6834", "XGBoost (tuned)": "#1baf7a"}
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
RISK_COLORS = ["#86b6ef", "#2a78d6", "#104281"]           # Low, Medium, High (ordinal ramp)
SEQ_RANGE = ["#cde2fb", "#0d366b"]
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

model_scale = alt.Scale(domain=MODELS, range=[MODEL_COLORS[m] for m in MODELS])
risk_scale = alt.Scale(domain=RISK, range=RISK_COLORS)


def _style(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(labelColor=INK2, titleColor=INK2, gridColor=GRID, domainColor=GRID,
                            tickColor=GRID, labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=INK2, titleColor=INK2, labelFontSize=12, titleFontSize=12,
                              orient="top", symbolSize=140)
            .configure_title(color=INK, fontSize=14, anchor="start", fontWeight="normal"))


# ---------------------------------------------------------------- confusion matrix
def confusion_heatmap(cm, normalize=False, title=""):
    cm = np.array(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) * 100 if normalize else cm
    rows = [{"Actual": RISK[i], "Predicted": RISK[j], "value": shown[i, j], "count": int(cm[i, j]),
             "pct": cm[i, j] / cm[i].sum() * 100,
             "label": f"{shown[i, j]:.1f}%" if normalize else f"{int(cm[i, j]):,}"}
            for i in range(3) for j in range(3)]
    df = pd.DataFrame(rows)
    vmax = 100 if normalize else float(cm.max())
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:O", sort=RISK, axis=alt.Axis(labelAngle=0, orient="bottom", title="Predicted risk group")),
        y=alt.Y("Actual:O", sort=RISK, axis=alt.Axis(title="Actual risk group")))
    rect = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("value:Q", scale=alt.Scale(domain=[0, vmax], range=SEQ_RANGE), legend=None),
        tooltip=["Actual", "Predicted", alt.Tooltip("count:Q", title="Records", format=","),
                 alt.Tooltip("pct:Q", title="% of actual class", format=".1f")])
    text = base.mark_text(fontSize=15, fontWeight="bold").encode(
        text="label:N",
        color=alt.condition(alt.datum.value > vmax * 0.5, alt.value("white"), alt.value(INK)))
    return _style((rect + text).properties(width=340, height=300, title=title))


# ---------------------------------------------------------------- model comparison
def _dodged_dots(long, row, value, x_title, domain, label_fmt, height):
    base = alt.Chart(long).encode(
        y=alt.Y(f"{row}:N", sort=list(dict.fromkeys(long[row])), title=None,
                axis=alt.Axis(labelPadding=8, domain=False, ticks=False)),
        yOffset=alt.YOffset("Model:N", sort=MODELS, scale=alt.Scale(paddingOuter=0.35)),
        x=alt.X(f"{value}:Q", scale=alt.Scale(domain=domain), axis=alt.Axis(tickCount=6), title=x_title),
        color=alt.Color("Model:N", scale=model_scale, sort=MODELS),
        tooltip=["Model", row, alt.Tooltip(f"{value}:Q", format=".3f")])
    dots = base.mark_circle(size=150, opacity=1)
    labels = base.mark_text(align="left", dx=11, fontSize=11, color=INK2).encode(
        text=alt.Text(f"{value}:Q", format=label_fmt))
    return _style((dots + labels).properties(height=height))


def metric_dots(comp: pd.DataFrame):
    # weighted recall equals accuracy by definition, so it is left out of the chart (it stays in the table)
    comp = comp[[c for c in comp.columns if not c.startswith("Recall")]]
    long = comp.melt("Model", var_name="Metric", value_name="Value")
    lo, hi = np.floor(long["Value"].min() - 1), np.ceil(long["Value"].max() + 1.5)
    return _dodged_dots(long, "Metric", "Value", "Score (%)  -  axis zoomed to the range of the results",
                        [lo, hi], ".2f", 95 * (comp.shape[1] - 1))


def per_class_dots(metric: str, tables: dict):
    long = pd.concat([t.assign(Model=m) for m, t in tables.items()])[["Model", "Class", metric]]
    lo = max(0, np.floor(long[metric].min() * 20 - 1) / 20)
    return _dodged_dots(long, "Class", metric, f"{metric}  -  axis starts at {lo:.2f}",
                        [lo, 1.0], ".2f", 300)


def error_bars(err: pd.DataFrame):
    base = alt.Chart(err).encode(
        x=alt.X("Error:N", sort=ERROR_ORDER, axis=alt.Axis(labelAngle=-20, title="Misclassification (actual → predicted)")),
        xOffset=alt.XOffset("Model:N", sort=MODELS),
        y=alt.Y("Count:Q", title="Test records"),
        color=alt.Color("Model:N", scale=model_scale, sort=MODELS),
        tooltip=["Model", "Error", alt.Tooltip("Count:Q", format=",")])
    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    labels = base.mark_text(dy=-6, fontSize=10, color=INK2).encode(text=alt.Text("Count:Q", format=","))
    return _style((bars + labels).properties(height=330))


def importance_bars(df: pd.DataFrame, value="Importance", color="#2a78d6", title=""):
    base = alt.Chart(df).encode(
        y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        x=alt.X(f"{value}:Q", title="Importance score"),
        tooltip=["Feature", alt.Tooltip(f"{value}:Q", format=".4f")])
    bars = base.mark_bar(color=color, cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
    labels = base.mark_text(align="left", dx=4, fontSize=11, color=INK2).encode(text=alt.Text(f"{value}:Q", format=".3f"))
    return _style((bars + labels).properties(height=max(260, 26 * len(df)), title=title))


# ---------------------------------------------------------------- WEKA
def cluster_bars(df: pd.DataFrame):
    base = alt.Chart(df).encode(
        x=alt.X("Cluster:N", axis=alt.Axis(labelAngle=0), title=None),
        y=alt.Y("Records:Q", title="Records"),
        tooltip=["Cluster", alt.Tooltip("Records:Q", format=","), alt.Tooltip("Percentage:Q", title="Share (%)")])
    bars = base.mark_bar(color="#2a78d6", size=70, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    labels = base.mark_text(dy=-8, fontSize=12, color=INK2).encode(
        text=alt.Text("lab:N")).transform_calculate(lab="format(datum.Records, ',') + '  (' + datum.Percentage + '%)'")
    return _style((bars + labels).properties(height=300))


def centroid_dots(cent: pd.DataFrame):
    long = cent.melt("Feature", var_name="Cluster", value_name="Value")
    base = alt.Chart(long).encode(
        y=alt.Y("Feature:N", sort=list(cent["Feature"]), title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("Value:Q", scale=alt.Scale(domain=[0, 0.55]), title="Normalised centroid value (0-1)"),
        color=alt.Color("Cluster:N", scale=alt.Scale(domain=["Cluster 0", "Cluster 1", "Cluster 2"], range=CLUSTER_COLORS)),
        tooltip=["Feature", "Cluster", alt.Tooltip("Value:Q", format=".4f")])
    rule = alt.Chart(long).mark_rule(color=GRID, strokeWidth=3).encode(
        y=alt.Y("Feature:N", sort=list(cent["Feature"])), x="min(Value):Q", x2="max(Value):Q")
    return _style((rule + base.mark_circle(size=150, opacity=1)).properties(height=300))


# ---------------------------------------------------------------- predictor
def probability_bars(proba: dict):
    """proba: {model name: {class: probability 0-1}}  (one or several models)."""
    df = pd.DataFrame([{"Model": m, "Class": c, "Probability": p * 100} for m, d in proba.items() for c, p in d.items()])
    multi = len(proba) > 1
    enc = dict(
        x=alt.X("Class:N", sort=RISK, axis=alt.Axis(labelAngle=0), title=None),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 100]), title="Estimated probability (%)"),
        tooltip=["Model", "Class", alt.Tooltip("Probability:Q", format=".2f")])
    if multi:
        enc["xOffset"] = alt.XOffset("Model:N", sort=MODELS)
        enc["color"] = alt.Color("Model:N", scale=model_scale, sort=MODELS)
    else:
        enc["color"] = alt.Color("Class:N", scale=risk_scale, legend=None)
    base = alt.Chart(df).encode(**enc)
    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    labels = base.mark_text(dy=-7, fontSize=11, color=INK2).encode(text=alt.Text("Probability:Q", format=".1f"))
    return _style((bars + labels).properties(height=300))


def whatif_lines(df: pd.DataFrame, feature: str, current):
    base = alt.Chart(df).encode(
        x=alt.X("value:Q", title=feature, scale=alt.Scale(zero=False)),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 100]), title="Estimated probability (%)"),
        color=alt.Color("Class:N", scale=risk_scale, sort=RISK),
        tooltip=[alt.Tooltip("value:Q", title=feature, format=".2f"), "Class", alt.Tooltip("Probability:Q", format=".1f")])
    line = base.mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=45, filled=True))
    now = alt.Chart(pd.DataFrame({"value": [current]})).mark_rule(color=INK2, strokeDash=[4, 4]).encode(x="value:Q")
    return _style((line + now).properties(height=300))


# ---------------------------------------------------------------- data explorer
def risk_count_bars(counts: pd.DataFrame):
    base = alt.Chart(counts).encode(
        x=alt.X("Risk:N", sort=RISK, axis=alt.Axis(labelAngle=0), title=None),
        y=alt.Y("Records:Q"), color=alt.Color("Risk:N", scale=risk_scale, legend=None),
        tooltip=["Risk", alt.Tooltip("Records:Q", format=",")])
    bars = base.mark_bar(size=80, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    labels = base.mark_text(dy=-8, fontSize=12, color=INK2).encode(text=alt.Text("Records:Q", format=","))
    return _style((bars + labels).properties(height=280))


def density_by_risk(hist: pd.DataFrame, feature: str):
    base = alt.Chart(hist).encode(
        x=alt.X("x:Q", title=feature, scale=alt.Scale(zero=False)),
        y=alt.Y("density:Q", title="Share of group (density)", axis=alt.Axis(format="%", labelOverlap=True)),
        color=alt.Color("Risk:N", scale=risk_scale, sort=RISK),
        tooltip=["Risk", alt.Tooltip("x:Q", format=".2f", title=feature), alt.Tooltip("density:Q", format=".3f")])
    area = base.mark_area(opacity=0.15, line=False).encode(color=alt.Color("Risk:N", scale=risk_scale, legend=None))
    return _style((area + base.mark_line(strokeWidth=2.5)).properties(height=300))


def box_by_risk(sample: pd.DataFrame, feature: str):
    ch = alt.Chart(sample).mark_boxplot(extent=1.5, size=55, median=alt.MarkConfig(color="white")).encode(
        x=alt.X("Risk:N", sort=RISK, axis=alt.Axis(labelAngle=0), title=None),
        y=alt.Y(f"{feature}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Risk:N", scale=risk_scale, legend=None))
    return _style(ch.properties(height=300))


def scatter_by_risk(sample: pd.DataFrame, x: str, y: str):
    ch = alt.Chart(sample).mark_circle(size=22, opacity=0.45).encode(
        x=alt.X(f"{x}:Q", scale=alt.Scale(zero=False)), y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Risk:N", scale=risk_scale, sort=RISK),
        tooltip=[x, y, "Risk"])
    return _style(ch.properties(height=360))


def correlation_heatmap(corr: pd.DataFrame):
    long = corr.reset_index().melt("index", var_name="b", value_name="r").rename(columns={"index": "a"})
    order = list(corr.columns)
    ch = alt.Chart(long).mark_rect().encode(
        x=alt.X("b:N", sort=order, axis=alt.Axis(labelAngle=-60, title=None, labelLimit=160)),
        y=alt.Y("a:N", sort=order, axis=alt.Axis(title=None, labelLimit=160)),
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 0, 1], range=DIVERGING), title="Correlation"),
        tooltip=["a", "b", alt.Tooltip("r:Q", format=".2f")])
    return _style(ch.properties(height=520))


def target_corr_bars(series: pd.Series, target: str):
    df = series.drop(target).sort_values().reset_index()
    df.columns = ["Feature", "r"]
    df["Sign"] = np.where(df["r"] >= 0, "Positive", "Negative")
    order = list(df.sort_values("r", ascending=False)["Feature"])
    base = alt.Chart(df).encode(
        y=alt.Y("Feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        x=alt.X("r:Q", scale=alt.Scale(domain=[-1, 1]), title=f"Correlation with {target}"),
        color=alt.Color("Sign:N", scale=alt.Scale(domain=["Positive", "Negative"], range=["#e34948", "#2a78d6"]),
                        legend=alt.Legend(title=None)),
        tooltip=["Feature", alt.Tooltip("r:Q", format=".3f")])
    bars = base.mark_bar()
    pos = base.transform_filter("datum.r >= 0").mark_text(align="left", dx=4, fontSize=10, color=INK2).encode(text=alt.Text("r:Q", format=".2f"))
    neg = base.transform_filter("datum.r < 0").mark_text(align="right", dx=-4, fontSize=10, color=INK2).encode(text=alt.Text("r:Q", format=".2f"))
    labels = pos + neg
    return _style((bars + labels).properties(height=max(300, 24 * len(df))))
