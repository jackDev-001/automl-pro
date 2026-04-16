import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import io
import json
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
for key, default in {
    "logged_in": False,
    "model": None,
    "X_columns": None,
    "problem_type": None,
    "best_score": None,
    "all_scores": None,
    "df": None,
    "target": None,
    "train_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #04040a;
    --surface:   #0d0d1a;
    --border:    rgba(120,120,255,0.15);
    --accent:    #7c6dff;
    --accent2:   #00e5ff;
    --accent3:   #ff4ecd;
    --text:      #e8e8f0;
    --muted:     #6b6b8a;
    --success:   #00e5a0;
    --warn:      #ffd166;
    --danger:    #ff4e6a;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(124,109,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,229,255,0.10) 0%, transparent 55%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 22px;
    backdrop-filter: blur(12px);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--accent2) !important; font-size: 28px !important; font-weight: 800; font-family: 'JetBrains Mono', monospace !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5a4dff);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 0.6em 1.4em;
    transition: all 0.25s ease;
    box-shadow: 0 4px 24px rgba(124,109,255,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(124,109,255,0.4);
}

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 14px; padding: 4px; gap: 4px; }
[data-baseweb="tab"] { border-radius: 10px !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; color: var(--muted) !important; }
[aria-selected="true"] { background: rgba(124,109,255,0.25) !important; color: var(--accent) !important; }

/* ── Inputs ── */
.stTextInput input, .stSelectbox select {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(16px);
    animation: fadeUp 0.5s ease both;
}
.card-accent { border-color: rgba(124,109,255,0.4); }
.card-success { border-color: rgba(0,229,160,0.3); }

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-purple { background: rgba(124,109,255,0.2); color: var(--accent); border: 1px solid rgba(124,109,255,0.3); }
.badge-cyan   { background: rgba(0,229,255,0.15); color: var(--accent2); border: 1px solid rgba(0,229,255,0.3); }
.badge-green  { background: rgba(0,229,160,0.15); color: var(--success); border: 1px solid rgba(0,229,160,0.3); }
.badge-pink   { background: rgba(255,78,205,0.15); color: var(--accent3); border: 1px solid rgba(255,78,205,0.3); }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Big Title ── */
.hero-title {
    font-size: 52px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #fff 0%, var(--accent) 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 4px;
}
.hero-sub {
    color: var(--muted);
    font-size: 14px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section heading ── */
.sec-heading {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Stat row ── */
.stat-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px;
}
.stat-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--text);
}
.stat-pill span { color: var(--accent2); font-weight: 600; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.pulse { animation: pulse 2s infinite; }

/* ── Login page ── */
.login-wrap {
    max-width: 420px;
    margin: 80px auto;
    padding: 40px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 24px;
    backdrop-filter: blur(20px);
    animation: fadeUp 0.6s ease both;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,109,255,0.3); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────
USERS = {"admin": "1234", "jack": "empire"}

if not st.session_state.logged_in:
    st.markdown("""
    <div class="login-wrap">
      <div class="hero-title" style="font-size:34px;">⚡ AutoML Pro</div>
      <p class="hero-sub" style="margin-bottom:28px;">Authenticate to continue</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("#### 🔐 Sign In")
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••")
        if st.button("Continue →", use_container_width=True):
            if USERS.get(username) == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials. Try admin / 1234")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px">
      <div class="hero-title" style="font-size:26px;">⚡ AutoML</div>
      <div class="hero-sub" style="font-size:10px;">Pro Edition</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    user = st.session_state.get("username", "admin")
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
      <div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#7c6dff,#00e5ff);
                  display:flex;align-items:center;justify-content:center;font-weight:800;font-size:14px;">
        {user[0].upper()}
      </div>
      <div>
        <div style="font-weight:700;font-size:14px;">{user.title()}</div>
        <div class="badge badge-green">Active</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    page = st.radio("", ["🏠 Dashboard", "🔬 EDA", "🤖 Train", "🎯 Predict", "📋 History", "⚙️ Settings"],
                    label_visibility="collapsed")

    st.markdown("---")

    # Model status
    if st.session_state.model:
        st.markdown(f"""
        <div class="card card-success" style="padding:16px;">
          <div class="sec-heading" style="margin-bottom:8px;">✅ Model Ready</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--muted);">
            Type: <span style="color:var(--accent2);">{st.session_state.problem_type}</span><br>
            Score: <span style="color:var(--success);">{round(st.session_state.best_score,4)}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section(icon, title):
    st.markdown(f'<div class="sec-heading">{icon} {title}</div>', unsafe_allow_html=True)

def card_open(extra_class=""):
    st.markdown(f'<div class="card {extra_class}">', unsafe_allow_html=True)

def card_close():
    st.markdown('</div>', unsafe_allow_html=True)

def eda_stats(df):
    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    numerics = int(df.select_dtypes(include=np.number).shape[1])
    return rows, cols, missing, numerics

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown('<div class="hero-title">AutoML Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">No-code intelligent model builder • Fast • Automatic • Powerful</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", len(st.session_state.train_history), "Today")
    c2.metric("Model Ready", "Yes" if st.session_state.model else "No")
    c3.metric("Dataset", f"{st.session_state.df.shape[0]} rows" if st.session_state.df is not None else "—")
    c4.metric("Best Score", round(st.session_state.best_score, 4) if st.session_state.best_score else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    c_a, c_b, c_c = st.columns(3)
    with c_a:
        st.markdown("""
        <div class="card card-accent" style="min-height:160px;">
          <div style="font-size:32px;">📂</div>
          <div style="font-weight:700;font-size:16px;margin:8px 0 4px;">Upload Data</div>
          <div style="color:var(--muted);font-size:13px;">Go to Train tab, upload your CSV and let AutoML do the rest.</div>
        </div>""", unsafe_allow_html=True)
    with c_b:
        st.markdown("""
        <div class="card" style="min-height:160px;border-color:rgba(0,229,255,0.3);">
          <div style="font-size:32px;">🤖</div>
          <div style="font-weight:700;font-size:16px;margin:8px 0 4px;">Auto Train</div>
          <div style="color:var(--muted);font-size:13px;">Multiple models trained, compared, and best one selected automatically.</div>
        </div>""", unsafe_allow_html=True)
    with c_c:
        st.markdown("""
        <div class="card" style="min-height:160px;border-color:rgba(255,78,205,0.3);">
          <div style="font-size:32px;">🎯</div>
          <div style="font-weight:700;font-size:16px;margin:8px 0 4px;">Predict</div>
          <div style="color:var(--muted);font-size:13px;">Run live predictions on new data using your trained model.</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────
elif page == "🔬 EDA":
    section("🔬", "Exploratory Data Analysis")

    if st.session_state.df is None:
        st.info("📂 Upload a dataset first in the **Train** tab.")
    else:
        df = st.session_state.df
        rows, cols, missing, numerics = eda_stats(df)

        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-pill">Rows: <span>{rows:,}</span></div>
          <div class="stat-pill">Columns: <span>{cols}</span></div>
          <div class="stat-pill">Missing Values: <span>{missing}</span></div>
          <div class="stat-pill">Numeric Cols: <span>{numerics}</span></div>
        </div>
        """, unsafe_allow_html=True)

        tab_ov, tab_corr, tab_dist, tab_miss = st.tabs(["📊 Overview", "🔗 Correlation", "📈 Distribution", "❓ Missing"])

        with tab_ov:
            st.markdown("**Sample Data**")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("**Descriptive Statistics**")
            st.dataframe(df.describe(), use_container_width=True)
            st.markdown("**Data Types**")
            dtype_df = pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.values.astype(str),
                                     "Unique": [df[c].nunique() for c in df.columns],
                                     "Nulls": df.isnull().sum().values})
            st.dataframe(dtype_df, use_container_width=True)

        with tab_corr:
            num_df = df.select_dtypes(include=np.number)
            if num_df.shape[1] < 2:
                st.warning("Need at least 2 numeric columns for correlation.")
            else:
                corr = num_df.corr()
                st.markdown("**Pearson Correlation Matrix**")
                st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}"),
                             use_container_width=True)

        with tab_dist:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                sel = st.selectbox("Select Column", num_cols)
                col_data = df[sel].dropna()
                hist_df = pd.cut(col_data, bins=30).value_counts().sort_index()
                hist_series = pd.Series(hist_df.values,
                                        index=[str(i.mid.round(2)) for i in hist_df.index],
                                        name=sel)
                st.bar_chart(hist_series)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", round(col_data.mean(), 3))
                c2.metric("Std", round(col_data.std(), 3))
                c3.metric("Min", round(col_data.min(), 3))
                c4.metric("Max", round(col_data.max(), 3))
            else:
                st.warning("No numeric columns found.")

        with tab_miss:
            miss_df = df.isnull().sum().reset_index()
            miss_df.columns = ["Column", "Missing Count"]
            miss_df["% Missing"] = (miss_df["Missing Count"] / len(df) * 100).round(2)
            miss_df = miss_df[miss_df["Missing Count"] > 0]
            if miss_df.empty:
                st.success("🎉 No missing values found!")
            else:
                st.dataframe(miss_df, use_container_width=True)
                st.bar_chart(miss_df.set_index("Column")["Missing Count"])

# ─────────────────────────────────────────────
# PAGE: TRAIN
# ─────────────────────────────────────────────
elif page == "🤖 Train":
    section("🤖", "Model Training")

    card_open("card-accent")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Problem Description**")
        user_problem = st.text_input("Describe your ML goal", placeholder="e.g. predict house price, classify spam email")
    with col_b:
        st.markdown("**Dataset**")
        file = st.file_uploader("Upload CSV", type=["csv"])
    card_close()

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        rows, cols, missing, numerics = eda_stats(df)
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-pill">Rows: <span>{rows:,}</span></div>
          <div class="stat-pill">Columns: <span>{cols}</span></div>
          <div class="stat-pill">Missing: <span>{missing}</span></div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

    if st.session_state.df is not None and user_problem:
        df = st.session_state.df
        card_open()
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("🎯 Target Column", df.columns)
        with col2:
            test_size = st.slider("Test Split %", 10, 40, 20)
        card_close()

        if st.button("⚡ Launch AutoML Training", use_container_width=True):
            progress_bar = st.progress(0)
            status = st.empty()

            stages = [
                (10, "🔍 Analyzing dataset..."),
                (25, "🧹 Preprocessing features..."),
                (40, "📐 Detecting problem type..."),
                (60, "🤖 Training model ensemble..."),
                (80, "🏆 Selecting best model..."),
                (95, "📊 Computing metrics..."),
                (100, "✅ Done!"),
            ]

            for pct, msg in stages:
                for p in range(progress_bar._percent if hasattr(progress_bar, "_percent") else 0, pct):
                    time.sleep(0.015)
                    progress_bar.progress(p + 1)
                status.markdown(f'<div class="pulse" style="color:var(--accent2);font-family:\'JetBrains Mono\',monospace;font-size:13px;">{msg}</div>',
                                unsafe_allow_html=True)

            try:
                from automl_engine import auto_ml
                model, best_score, all_scores, problem, X = auto_ml(df, target, user_problem)

                st.session_state.model = model
                st.session_state.X_columns = X.columns.tolist()
                st.session_state.problem_type = problem
                st.session_state.best_score = best_score
                st.session_state.all_scores = all_scores
                st.session_state.target = target
                st.session_state.train_history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "problem": problem,
                    "score": round(best_score, 4),
                    "target": target,
                    "rows": len(df),
                })

                status.empty()
                st.success("🎉 Model trained successfully!")

                c1, c2, c3 = st.columns(3)
                c1.metric("Problem Type", problem)
                c2.metric("Best Score", round(best_score, 4))
                c3.metric("Features Used", len(X.columns))

                st.markdown("**📊 Model Comparison**")
                scores_df = pd.DataFrame(list(all_scores.items()), columns=["Model", "Score"]).set_index("Model")
                st.bar_chart(scores_df)

                if hasattr(model, "feature_importances_"):
                    st.markdown("**📌 Feature Importance**")
                    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.bar_chart(fi)

                # Download model
                buf = io.BytesIO()
                pickle.dump(model, buf)
                buf.seek(0)
                st.download_button("⬇️ Download Model (.pkl)", data=buf, file_name="automl_model.pkl",
                                   mime="application/octet-stream")

                # Download model report
                report = {
                    "trained_at": datetime.now().isoformat(),
                    "target": target,
                    "problem_type": problem,
                    "best_score": round(best_score, 4),
                    "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
                    "features": X.columns.tolist(),
                    "dataset_shape": list(df.shape),
                }
                st.download_button("📋 Download Report (JSON)", data=json.dumps(report, indent=2),
                                   file_name="automl_report.json", mime="application/json")

            except ImportError:
                status.error("⚠️ `automl_engine.py` not found. Make sure it's in the same directory.")
            except Exception as e:
                status.error(f"Training failed: {e}")

# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────
elif page == "🎯 Predict":
    section("🎯", "Live Prediction")

    if st.session_state.model is None:
        st.warning("⚠️ No model trained yet. Go to **Train** tab first.")
    else:
        model = st.session_state.model
        X_cols = st.session_state.X_columns
        df_ref = st.session_state.df

        tab_manual, tab_batch = st.tabs(["✍️ Manual Input", "📂 Batch CSV"])

        with tab_manual:
            st.markdown("Fill in values for each feature:")
            input_data = {}
            cols_per_row = 3
            col_groups = [X_cols[i:i+cols_per_row] for i in range(0, len(X_cols), cols_per_row)]

            for group in col_groups:
                row_cols = st.columns(len(group))
                for col, feat in zip(row_cols, group):
                    with col:
                        if df_ref is not None and feat in df_ref.columns:
                            dtype = df_ref[feat].dtype
                            if dtype in [np.float64, np.int64]:
                                mn = float(df_ref[feat].min())
                                mx = float(df_ref[feat].max())
                                med = float(df_ref[feat].median())
                                input_data[feat] = st.number_input(feat, min_value=mn, max_value=mx, value=med)
                            else:
                                opts = df_ref[feat].dropna().unique().tolist()[:50]
                                input_data[feat] = st.selectbox(feat, opts)
                        else:
                            input_data[feat] = st.text_input(feat)

            if st.button("⚡ Predict", use_container_width=True):
                try:
                    row = pd.DataFrame([input_data])
                    pred = model.predict(row)[0]
                    st.markdown(f"""
                    <div class="card card-success" style="text-align:center;padding:32px;">
                      <div style="font-size:13px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Prediction Result</div>
                      <div style="font-size:48px;font-weight:800;color:var(--success);font-family:'JetBrains Mono',monospace;">{pred}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(row)[0]
                        classes = model.classes_
                        proba_df = pd.DataFrame({"Class": classes, "Probability": proba}).set_index("Class")
                        st.markdown("**Class Probabilities**")
                        st.bar_chart(proba_df)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        with tab_batch:
            batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                st.dataframe(batch_df.head(), use_container_width=True)
                if st.button("⚡ Predict All", use_container_width=True):
                    try:
                        common = [c for c in X_cols if c in batch_df.columns]
                        preds = model.predict(batch_df[common])
                        batch_df["🎯 Prediction"] = preds
                        st.success(f"Predicted {len(preds)} rows!")
                        st.dataframe(batch_df, use_container_width=True)
                        csv_out = batch_df.to_csv(index=False).encode()
                        st.download_button("⬇️ Download Predictions", data=csv_out,
                                           file_name="predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Batch prediction error: {e}")

# ─────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────
elif page == "📋 History":
    section("📋", "Training History")

    if not st.session_state.train_history:
        st.info("No training runs yet. Go to **Train** tab.")
    else:
        hist_df = pd.DataFrame(st.session_state.train_history)
        st.dataframe(hist_df, use_container_width=True)
        st.bar_chart(hist_df.set_index("time")["score"])

        if st.button("🗑️ Clear History"):
            st.session_state.train_history = []
            st.rerun()

# ─────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────
elif page == "⚙️ Settings":
    section("⚙️", "Settings")

    card_open()
    st.markdown("**🎨 Display**")
    st.toggle("Compact Mode (Coming Soon)", value=False)
    st.slider("Chart Height", 200, 600, 350)
    card_close()

    card_open()
    st.markdown("**🤖 Model Config**")
    st.slider("Cross-validation Folds", 2, 10, 5)
    st.multiselect("Enabled Algorithms",
                   ["Random Forest", "XGBoost", "Gradient Boosting", "SVM", "KNN", "Logistic Regression"],
                   default=["Random Forest", "Gradient Boosting"])
    st.selectbox("Scoring Metric (Classification)", ["accuracy", "f1", "roc_auc", "precision", "recall"])
    st.selectbox("Scoring Metric (Regression)", ["r2", "neg_rmse", "neg_mae"])
    card_close()

    card_open()
    st.markdown("**💾 Session**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Trained Model"):
            st.session_state.model = None
            st.session_state.best_score = None
            st.session_state.all_scores = None
            st.success("Model cleared.")
    with col2:
        if st.button("🔄 Reset Everything"):
            for k in list(st.session_state.keys()):
                if k != "logged_in":
                    del st.session_state[k]
            st.rerun()
    card_close()