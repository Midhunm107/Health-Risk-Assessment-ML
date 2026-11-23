"""
Health Risk Assessment System - project presentation app (Streamlit).

Run (from the repository root):   streamlit run app/streamlit_app.py
Needs: models/*.joblib, models/meta.json, models/eval.json
       (create them once with `python -m src.train_models data/health_risk_predictor.csv`,
        or click "Train the three models now" on the Live Risk Predictor page)
Optional: put the dataset at data/health_risk_predictor.csv (or upload it in the app)
          to switch on the "Data Explorer" dashboard.
Every page except the Live Risk Predictor and Data Explorer works without either.
"""
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent          # app/
ROOT = BASE.parent                              # repository root
sys.path.insert(0, str(ROOT))                   # lets the app import the shared `src` package

import charts as ch  # noqa: E402
import results_data as rd  # noqa: E402
from src.prediction import class_probs  # noqa: E402

ASSETS = BASE / "assets"
MODELS = ROOT / "models"
DATA_FILE = ROOT / "data" / "health_risk_predictor.csv"
RISK = rd.RISK

st.set_page_config(page_title="Health Risk Assessment System", layout="wide")

DISCLAIMER = (
    "Educational demonstration only. The models are trained on a public Kaggle dataset and are not "
    "a medical device or a substitute for professional medical advice."
)
MODEL_FILES = {"Decision Tree": "dt_model.joblib", "Random Forest": "rf_model.joblib",
               "XGBoost (tuned)": "xgb_model.joblib"}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def img(name: str, caption: str = ""):
    p = ASSETS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Image not found: assets/{name}")


def table(df: pd.DataFrame, formats: dict | None = None):
    data = df.style.format(formats) if formats else df
    st.dataframe(data, hide_index=True, width="stretch")


def chart_with_table(chart, df: pd.DataFrame, formats: dict | None = None):
    t1, t2 = st.tabs(["Chart", "Table"])
    with t1:
        st.altair_chart(chart, width="stretch")
    with t2:
        table(df, formats)


def section(title: str, text: str = ""):
    st.markdown(f"#### {title}")
    if text:
        st.caption(text)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_overview():
    st.title("Health Risk Assessment System")
    st.subheader("Exploratory Clustering with WEKA and Heart Disease Risk Prediction Using Machine Learning")
    st.caption("TKM College of Engineering, Kollam | Department of Computer Applications | Machine Learning (23MCAJ301)")

    st.markdown(
        "**Problem.** Given demographic, lifestyle and clinical measurements, can we classify a "
        "person's heart disease risk as **Low, Medium or High** using only structured, "
        "easily-collected data - with no imaging or continuous monitoring?"
    )
    best = rd.METRICS["XGBoost (tuned)"]["accuracy"] * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", "97,297")
    c2.metric("Input features", "28")
    c3.metric("Best accuracy (XGBoost)", f"{best:.2f}%")
    c4.metric("WEKA clusters (k)", "3")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Two complementary tracks")
        st.markdown(
            "**Exploratory track (unsupervised)**\n\n"
            "- WEKA SimpleKMeans, k = 3 on 28 normalised attributes\n"
            "- Finds natural groupings without using the risk label\n"
            "- Interprets clusters through centroid comparison\n\n"
            "**Supervised track (classification)**\n\n"
            "- Decision Tree, Random Forest and XGBoost\n"
            "- One shared leakage-free pipeline and 80:20 stratified split\n"
            "- Evaluated by accuracy, precision, recall, F1 and confusion matrices"
        )
    with right:
        st.markdown("### Accuracy at a glance")
        comp = rd.comparison_table()[["Model", "Accuracy"]]
        st.altair_chart(ch.metric_dots(comp), width="stretch")


def page_dataset():
    st.title("Dataset and Preprocessing")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", "97,297")
    c2.metric("Columns", "33")
    c3.metric("Missing / duplicates", "0 / 0")

    st.markdown("### Pipeline")
    st.markdown(
        "1. **Correlation analysis** - the hypertension risk score correlates r = 0.80 with the heart-disease risk score.\n"
        "2. **Leakage removal** - the 5 pre-computed risk-score columns (diabetes, hypertension, obesity, "
        "cholesterol, heart disease) are dropped so the models cannot simply read the answer.\n"
        "3. **Target creation** - the continuous heart-disease risk score is split into 3 balanced classes by "
        "equal-frequency binning."
    )
    st.code('pd.qcut(df["heart_disease_risk_score"], q=3, labels=["Low", "Medium", "High"])', language="python")
    st.markdown(
        "4. **Encoding** - 6 categorical columns one-hot encoded (28 features expand to 40).\n"
        "5. **Split** - stratified 80:20 train/test, `random_state = 42` (77,837 train / 19,460 test records)."
    )
    img("correlation_heatmap.png", "Correlation heatmap used to identify leakage (see Data Explorer for an interactive version)")


def page_architecture():
    st.title("System Architecture")
    img("architecture.png", "End-to-end architecture: shared preprocessing, WEKA exploratory track, supervised classification track")


def page_results():
    st.title("Model Results")
    comp = rd.comparison_table()
    table(comp, {c: "{:.2f}%" for c in comp.columns[1:]})
    st.caption("Weighted averages on the shared held-out test set of 19,460 records, computed from each model's confusion matrix.")

    st.markdown(
        "**Takeaway.** XGBoost is best on every metric (85.01% accuracy), ahead of Random Forest (83.54%) and the "
        "Decision Tree (82.37%). For all three models the *Medium* class is the hardest."
    )
    cols = st.columns(3)
    for col, m in zip(cols, rd.MODELS):
        with col:
            st.markdown(f"**{m}**  \n{rd.OWNERS[m]}")
            table(rd.METRICS[m]["per_class"][["Class", "Precision", "Recall", "F1-score"]],
                  {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"})

    with st.expander("XGBoost hyperparameter tuning"):
        st.markdown(
            "A baseline model was tuned with `RandomizedSearchCV` - 20 random candidates, 3-fold stratified "
            "cross-validation on the training set, macro-F1 scoring. The test set was not used during tuning."
        )
        table(rd.XGB_PARAMS)
        table(rd.XGB_BASE_VS_TUNED, {"Accuracy (%)": "{:.2f}", "Macro F1": "{:.4f}"})
        st.caption("Tuning moved accuracy by only 0.03 points: the baseline was already close to what these features allow.")
    with st.expander("Random Forest feature importance (notebook figure)"):
        img("rf_feature_importance.png", "Random Forest - top 15 predictors")
    with st.expander("Decision Tree structure"):
        img("dt_tree.png", "Decision Tree visualisation (top levels)")


def page_dashboard():
    st.title("Model Dashboard")
    st.caption("Interactive comparison of the three classifiers on the shared 19,460-record test set.")

    m = rd.METRICS
    x, r, d = m["XGBoost (tuned)"], m["Random Forest"], m["Decision Tree"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("XGBoost accuracy", f"{x['accuracy'] * 100:.2f}%", f"{(x['accuracy'] - r['accuracy']) * 100:+.2f} pts vs Random Forest")
    k2.metric("Random Forest accuracy", f"{r['accuracy'] * 100:.2f}%", f"{(r['accuracy'] - d['accuracy']) * 100:+.2f} pts vs Decision Tree")
    k3.metric("Decision Tree accuracy", f"{d['accuracy'] * 100:.2f}%")
    medium_gain = x["per_class"].loc[1, "F1-score"] - d["per_class"].loc[1, "F1-score"]
    k4.metric("Medium-class F1 (XGBoost)", f"{x['per_class'].loc[1, 'F1-score']:.2f}", f"{medium_gain:+.2f} vs Decision Tree")

    section("1. Overall metrics",
            "Weighted over the three classes. The axis is zoomed to the range of the results, so compare the gaps, not the bar lengths.")
    comp = rd.comparison_table()
    chart_with_table(ch.metric_dots(comp), comp, {c: "{:.2f}%" for c in comp.columns[1:]})

    section("2. Performance by risk class")
    metric = st.radio("Metric", ["F1-score", "Precision", "Recall"], horizontal=True, label_visibility="collapsed")
    tables = {name: m[name]["per_class"] for name in rd.MODELS}
    long = pd.concat([t.assign(Model=n) for n, t in tables.items()])[["Model", "Class", metric]]
    chart_with_table(ch.per_class_dots(metric, tables), long, {metric: "{:.3f}"})
    st.caption("Medium is the hardest class for every model: it sits between Low and High, so its boundaries overlap both neighbours.")

    section("3. Where do the errors happen?",
            "Misclassified test records by type. Almost all errors are between neighbouring risk bands.")
    err = rd.error_table()
    chart_with_table(ch.error_bars(err), err.pivot(index="Error", columns="Model", values="Count").reset_index(), None)
    far = err[err["Kind"] == "Low ↔ High"].groupby("Model")["Count"].sum()
    st.markdown(
        "Low ↔ High confusions out of 19,460 records: "
        + ", ".join(f"**{k}** {int(v)}" for k, v in far.reindex(rd.MODELS).items()) + "."
    )

    section("4. Confusion matrix explorer")
    c1, c2 = st.columns([1, 1])
    model = c1.selectbox("Model", rd.MODELS, index=2)
    mode = c2.radio("Show", ["Record counts", "% of each actual class"], horizontal=True)
    left, right = st.columns([1, 1])
    with left:
        st.altair_chart(ch.confusion_heatmap(rd.CONFUSION[model], normalize=mode.startswith("%"),
                                             title=f"{model} - confusion matrix"), width="stretch")
    with right:
        pc = rd.METRICS[model]["per_class"]
        table(pc, {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "Support": "{:,}"})
        st.caption("Rows are the actual class, columns the predicted class. The diagonal is correct predictions.")

    section("5. What the models rely on")
    tabs = st.tabs(["XGBoost (notebook, baseline model)", "Your trained models", "Random Forest (notebook figure)"])
    with tabs[0]:
        st.altair_chart(ch.importance_bars(rd.XGB_IMPORTANCE, title="Top 15 features by XGBoost importance"), width="stretch")
        st.caption(
            "Smoking status is one-hot encoded, so it appears as separate columns: Former (0.171) and Never (0.102) "
            "together are about as important as Age (0.277). Importance describes how the model used a feature; it does not "
            "show that the feature causes heart disease."
        )
    with tabs[1]:
        ev = load_eval()
        if not ev:
            st.info("Run `python -m src.train_models <csv>` to see importances and a re-check of the metrics from the models "
                    "trained on this machine.")
        else:
            which = st.radio("Model", [k for k, v in ev["models"].items() if v["importance"]], horizontal=True)
            imp = pd.DataFrame(ev["models"][which]["importance"])
            st.altair_chart(ch.importance_bars(imp, color=ch.MODEL_COLORS[which], title=f"{which} - local run"), width="stretch")
            chk = pd.DataFrame([{"Model": k, "Accuracy (local run)": v["accuracy"] * 100,
                                 "Accuracy (reported)": rd.METRICS[k]["accuracy"] * 100} for k, v in ev["models"].items()])
            table(chk, {"Accuracy (local run)": "{:.2f}%", "Accuracy (reported)": "{:.2f}%"})
    with tabs[2]:
        img("rf_feature_importance.png", "Random Forest - top 15 predictors of heart disease risk")


def page_weka():
    st.title("Exploratory Clustering with WEKA")
    left, right = st.columns([1, 1])
    with left:
        section("SimpleKMeans configuration")
        table(rd.WEKA_PARAMS)
    with right:
        section("Cluster sizes")
        st.altair_chart(ch.cluster_bars(rd.WEKA_DIST), width="stretch")

    section("Cluster centroids (selected features, normalised 0-1)",
            "Each row shows where the three cluster centres sit on one feature; the grey bar spans the gap between them.")
    chart_with_table(ch.centroid_dots(rd.WEKA_CENTROIDS), rd.WEKA_CENTROIDS,
                     {c: "{:.4f}" for c in rd.WEKA_CENTROIDS.columns[1:]})
    st.markdown(
        "**Reading the centroids.** Age, BMI and systolic BP barely differ between clusters. *Family history of "
        "diabetes* separates them clearly (0.34 vs 0.26 vs 0.11), with glucose and HbA1c following the same "
        "ordering. The clusters are therefore driven mainly by metabolic and family-history signals."
    )
    st.warning("Limitation: k = 3 mirrors the three risk classes; the clusters are exploratory, depend on the random seed, "
               "and are not labelled risk groups.")
    t1, t2, t3 = st.tabs(["Scatter plot", "Centroid table (WEKA)", "Cluster summary (WEKA)"])
    with t1:
        img("weka_scatter.png", "WEKA cluster scatter plot")
    with t2:
        img("weka_centroids_top.png", "WEKA clusterer output - centroids")
    with t3:
        img("weka_cluster_summary.png", "WEKA clusterer output - summary")


# ----------------------------------------------------------------------------
# Data Explorer
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_csv(source) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(source) if isinstance(source, bytes) else source)
    if "heart_disease_risk_score" in df.columns:
        df["Risk"] = pd.qcut(df["heart_disease_risk_score"], 3, labels=RISK).astype(str)
    return df


def density_table(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    lo, hi = df[feature].quantile([0.005, 0.995])
    if lo == hi:
        lo, hi = df[feature].min(), df[feature].max() + 1e-9
    rows = []
    for risk in RISK:
        v = df.loc[df["Risk"] == risk, feature].dropna().clip(lo, hi)
        if len(v) == 0:
            continue
        counts, edges = np.histogram(v, bins=40, range=(lo, hi))
        centres = (edges[:-1] + edges[1:]) / 2
        rows += [{"Risk": risk, "x": float(c), "density": float(n) / len(v)} for c, n in zip(centres, counts)]
    return pd.DataFrame(rows)


def page_explorer():
    st.title("Data Explorer")
    st.caption("Interactive exploration of the Health Risk Predictor dataset, coloured by the three risk classes.")

    source, label = None, ""
    if DATA_FILE.exists():
        source, label = str(DATA_FILE), f"data/{DATA_FILE.name}"
    up = st.file_uploader("Upload health_risk_predictor.csv" if source is None else "Or upload a different CSV", type="csv")
    if up is not None:
        source, label = up.getvalue(), up.name
    if source is None:
        st.info("Place the dataset at `data/health_risk_predictor.csv` or upload it above to enable this dashboard "
                "(the file is not bundled because it comes from Kaggle).")
        return
    df = load_csv(source)
    if "Risk" not in df.columns:
        st.error("This CSV has no `heart_disease_risk_score` column, so the risk classes cannot be built.")
        return

    # ---- filters
    f1, f2, f3 = st.columns([2, 2, 1])
    if "Age" in df.columns:
        lo, hi = int(df["Age"].min()), int(df["Age"].max())
        age = f1.slider("Age range", lo, hi, (lo, hi))
        df = df[df["Age"].between(*age)]
    if "gender" in df.columns:
        opts = sorted(df["gender"].dropna().unique())
        sel = f2.multiselect("Gender", opts, default=opts)
        df = df[df["gender"].isin(sel)]
    f3.metric("Records shown", f"{len(df):,}")
    if df.empty:
        st.warning("No records match the filters.")
        return

    numeric = [c for c in df.select_dtypes(include="number").columns if c not in rd.LEAKAGE_COLS]
    sample = df.sample(min(len(df), 6000), random_state=1)

    tab1, tab2, tab3, tab4 = st.tabs(["Risk groups", "Relationships", "Correlation and leakage", "Data preview"])
    with tab1:
        a, b = st.columns([1, 2])
        counts = df["Risk"].value_counts().reindex(RISK).fillna(0).astype(int).rename_axis("Risk").reset_index(name="Records")
        with a:
            section("Records per risk class", "Equal-frequency binning keeps the classes balanced.")
            st.altair_chart(ch.risk_count_bars(counts), width="stretch")
        with b:
            feat = st.selectbox("Feature", numeric, index=numeric.index("systolic_bp") if "systolic_bp" in numeric else 0, key="dist_feat")
            view = st.radio("View", ["Distribution", "Box plot"], horizontal=True)
            if view == "Distribution":
                st.altair_chart(ch.density_by_risk(density_table(df, feat), f"{feat}"), width="stretch")
                st.caption("Share of each risk group falling in each bin (40 bins between the 0.5th and 99.5th percentile).")
            else:
                st.altair_chart(ch.box_by_risk(sample[["Risk", feat]], feat), width="stretch")
                st.caption(f"Box plot on a random sample of {len(sample):,} records.")
        section("Average by risk class")
        means = df.groupby("Risk")[numeric].mean().reindex(RISK).T.reset_index().rename(columns={"index": "Feature"})
        table(means, {c: "{:.2f}" for c in RISK})
    with tab2:
        c1, c2 = st.columns(2)
        xf = c1.selectbox("X axis", numeric, index=numeric.index("Age") if "Age" in numeric else 0, key="sx")
        yf = c2.selectbox("Y axis", numeric, index=numeric.index("systolic_bp") if "systolic_bp" in numeric else 1, key="sy")
        st.altair_chart(ch.scatter_by_risk(sample[[xf, yf, "Risk"]], xf, yf), width="stretch")
        st.caption(f"Random sample of {len(sample):,} records. Heavy overlap between the classes explains why no single pair of "
                   "features separates them and why a model combining all 40 encoded features is needed.")
    with tab3:
        cols = [c for c in df.select_dtypes(include="number").columns]
        corr = df[cols].corr()
        target = "heart_disease_risk_score"
        if target in corr.columns:
            section("Correlation with the heart-disease risk score",
                    "The other pre-computed risk scores correlate strongly with the target. They were removed as data leakage.")
            st.altair_chart(ch.target_corr_bars(corr[target], target), width="stretch")
        with st.expander("Full correlation heatmap"):
            st.altair_chart(ch.correlation_heatmap(corr), width="stretch")
    with tab4:
        st.caption(f"Source: {label}")
        st.dataframe(df.head(200), width="stretch")
        st.dataframe(df[numeric].describe().T, width="stretch")


# ----------------------------------------------------------------------------
# Live predictor
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    meta = json.loads((MODELS / "meta.json").read_text())
    loaded = {}
    for label, fname in MODEL_FILES.items():
        if (MODELS / fname).exists():
            loaded[label] = joblib.load(MODELS / fname)
    return meta, loaded


def load_eval():
    p = MODELS / "eval.json"
    return json.loads(p.read_text()) if p.exists() else None


def pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _default_for(feat, meta):
    cats, nums, binary = meta["categorical"], meta["numeric"], meta["binary"]
    if feat in cats:
        return cats[feat][0]
    if feat in binary:
        return "No"
    s = nums[feat]
    return int(round(s["median"])) if s["is_int"] else float(s["median"])


def _init_state(meta):
    for feat in meta["feature_order"]:
        key = f"in_{feat}"
        if key not in st.session_state:
            st.session_state[key] = _default_for(feat, meta)


def load_sample(meta):
    for feat, val in rd.SAMPLE_PERSON.items():
        if feat not in meta["feature_order"]:
            continue
        key = f"in_{feat}"
        if feat in meta["categorical"]:
            if val in meta["categorical"][feat]:
                st.session_state[key] = val
        elif feat in meta["binary"]:
            st.session_state[key] = "Yes" if val else "No"
        else:
            s = meta["numeric"][feat]
            v = min(max(val, s["min"]), s["max"])
            st.session_state[key] = int(round(v)) if s["is_int"] else float(v)


def train_in_app():
    """Shown when no trained models exist: train them here, no command line needed."""
    st.warning("The models have not been trained yet. Train them once below (about 1-3 minutes) and the "
               "predictor will open automatically.")
    st.caption(f"This app looks for the models in: `{MODELS}`")
    present = sorted(p.name for p in MODELS.glob("*")) if MODELS.exists() else []
    st.caption("Files found there: " + (", ".join(present) if present else "none (the folder does not exist yet)"))

    csv_path = None
    if DATA_FILE.exists():
        st.success(f"Dataset found: `{DATA_FILE}`")
        csv_path = DATA_FILE
    up = st.file_uploader("Upload health_risk_predictor.csv", type="csv")
    if up is not None:
        DATA_FILE.parent.mkdir(exist_ok=True)
        DATA_FILE.write_bytes(up.getvalue())
        csv_path = DATA_FILE
    if csv_path is None:
        st.info("Upload the dataset CSV above (or put it at `data/health_risk_predictor.csv`).")
        return

    if st.button("Train the three models now", type="primary"):
        try:
            from src import train_models as tas
        except Exception as e:  # missing xgboost / sklearn etc.
            st.error(f"Could not import the training code: {e}\n\n"
                     "Install the requirements, then restart the app:  `python -m pip install -r requirements.txt`")
            return
        try:
            with st.spinner("Training Decision Tree, Random Forest and XGBoost on 77,837 rows..."):
                tas.main(str(csv_path), tune=False)
        except Exception as e:
            st.error(f"Training failed: {type(e).__name__}: {e}")
            st.exception(e)
            return
        load_models.clear()
        st.success("Done. Loading the predictor...")
        st.rerun()


def page_predictor():
    st.title("Live Risk Predictor")
    st.caption(DISCLAIMER)

    if not (MODELS / "meta.json").exists():
        train_in_app()
        return
    meta, models = load_models()
    if not models:
        st.error("meta.json found but no .joblib models in the models/ folder.")
        return
    _init_state(meta)

    top1, top2 = st.columns([2, 3])
    choice = top1.radio("Model", ["Compare all models"] + list(models.keys()), horizontal=False)
    with top2:
        st.button("Load the sample person from the report", on_click=load_sample, args=(meta,))
        st.caption("A 45-year-old non-smoking man with values in the usual range. In the project notebook the tuned XGBoost model "
                   "gave Low 55.92% / Medium 43.91% / High 0.17% for this person (Random Forest: 49% / 48% / 3%). "
                   "A model re-trained on your machine can differ slightly.")

    cats, nums, binary = meta["categorical"], meta["numeric"], meta["binary"]
    with st.form("predictor"):
        st.markdown("##### Demographics and lifestyle")
        cols = st.columns(3)
        i = 0
        for feat in meta["feature_order"]:
            if feat in cats:
                cols[i % 3].selectbox(pretty(feat), cats[feat], key=f"in_{feat}")
                i += 1
        for feat in meta["feature_order"]:
            if feat in binary:
                cols[i % 3].selectbox(pretty(feat), ["No", "Yes"], key=f"in_{feat}")
                i += 1
        st.markdown("##### Measurements")
        cols = st.columns(3)
        i = 0
        for feat in meta["feature_order"]:
            if feat in nums:
                s = nums[feat]
                if s["is_int"]:
                    cols[i % 3].number_input(pretty(feat), min_value=int(s["min"]), max_value=int(s["max"]),
                                             step=1, key=f"in_{feat}")
                else:
                    cols[i % 3].number_input(pretty(feat), min_value=float(s["min"]), max_value=float(s["max"]),
                                             step=0.1, format="%.2f", key=f"in_{feat}")
                i += 1
        submitted = st.form_submit_button("Predict risk")

    if submitted:
        values = {}
        for feat in meta["feature_order"]:
            v = st.session_state[f"in_{feat}"]
            values[feat] = (1 if v == "Yes" else 0) if feat in binary else v
        st.session_state["last_row"] = values
        st.session_state["last_choice"] = choice

    if "last_row" not in st.session_state:
        return
    row = pd.DataFrame([st.session_state["last_row"]])[meta["feature_order"]]
    use = list(models) if st.session_state["last_choice"] == "Compare all models" else [st.session_state["last_choice"]]
    use = [m for m in use if m in models]
    probs = {m: class_probs(models[m], row) for m in use}
    preds = {m: max(p, key=p.get) for m, p in probs.items()}

    st.markdown("### Result")
    if len(use) == 1:
        m = use[0]
        tone = {"Low": "success", "Medium": "warning", "High": "error"}[preds[m]]
        getattr(st, tone)(f"{m}: predicted heart disease risk **{preds[m]}**")
    else:
        cols = st.columns(len(use))
        for c, m in zip(cols, use):
            c.metric(m, preds[m])
            c.caption(f"{probs[m][preds[m]] * 100:.1f}% estimated probability")
        if len(set(preds.values())) == 1:
            st.success(f"All models agree: **{next(iter(preds.values()))}** risk.")
        else:
            st.warning("The models disagree - this person is probably near a boundary between two risk bands.")
    st.altair_chart(ch.probability_bars(probs), width="stretch")
    ptab = pd.DataFrame([{"Model": m, **{k: v * 100 for k, v in p.items()}} for m, p in probs.items()])
    with st.expander("Probability table"):
        table(ptab, {k: "{:.2f}%" for k in RISK})

    with st.expander("What-if explorer: change one measurement and watch the probabilities move"):
        numeric_feats = list(nums.keys())
        c1, c2 = st.columns(2)
        feat = c1.selectbox("Measurement to vary", numeric_feats,
                            index=numeric_feats.index("systolic_bp") if "systolic_bp" in numeric_feats else 0)
        wm = c2.selectbox("Model", list(models.keys()), index=len(models) - 1)
        s = nums[feat]
        sweep = np.linspace(s["min"], s["max"], 30)
        grid = pd.concat([row] * len(sweep), ignore_index=True)
        grid[feat] = np.round(sweep).astype(int) if s["is_int"] else sweep
        pr = models[wm].predict_proba(grid)
        names = [c if isinstance(c, str) else RISK[int(c)] for c in models[wm].classes_]
        long = pd.DataFrame([{"value": float(grid[feat].iloc[i]), "Class": names[j], "Probability": float(pr[i, j]) * 100}
                             for i in range(len(grid)) for j in range(len(names))])
        st.altair_chart(ch.whatif_lines(long, feat, float(row[feat].iloc[0])), width="stretch")
        st.caption("All other inputs stay as submitted. The dashed line marks the submitted value. This shows how the model "
                   "responds, not what would happen to a real person's health.")


def page_team():
    st.title("Team and Guidance")
    team = pd.DataFrame(
        [["Adithya Dev", "25MCA005", "Decision Tree"],
         ["Joyal John Mathew", "25MCA028", "XGBoost and WEKA clustering"],
         ["Midhun M. Pillai", "25MCA035", "Random Forest"]],
        columns=["Name", "Roll No.", "Contribution"])
    table(team)
    st.markdown("**Project guide:** Prof. Gopika Gopan  \n**Head of Department:** Prof. Natheera Beevi M")
    st.markdown("TKM College of Engineering, Kollam | Department of Computer Applications")


PAGES = {
    "Overview": page_overview,
    "Dataset and Preprocessing": page_dataset,
    "Architecture": page_architecture,
    "Model Results": page_results,
    "Model Dashboard": page_dashboard,
    "WEKA Clustering": page_weka,
    "Data Explorer": page_explorer,
    "Live Risk Predictor": page_predictor,
    "Team": page_team,
}

with st.sidebar:
    st.header("Health Risk Assessment")
    choice = st.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    present = st.toggle("Presentation mode", help="Hide Streamlit's toolbar and footer for a cleaner projector view.")

if present:
    st.markdown(
        "<style>#MainMenu, header, footer {visibility: hidden;} "
        ".block-container {padding-top: 1.5rem;}</style>",
        unsafe_allow_html=True,
    )

PAGES[choice]()
