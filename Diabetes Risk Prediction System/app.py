

import pickle
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go


st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #091320 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #091320 100%);
    border-right: 1px solid rgba(56,189,248,0.12);
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* ── Remove default Streamlit padding/header ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Hero title ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* ── Section headers ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    margin-top: 1.6rem;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(8px);
}
.metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1.1;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.2rem;
    letter-spacing: 0.06em;
}

/* ── Result banners ── */
.result-positive {
    background: linear-gradient(135deg, rgba(248,113,113,0.18) 0%, rgba(239,68,68,0.08) 100%);
    border: 1.5px solid rgba(248,113,113,0.5);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-negative {
    background: linear-gradient(135deg, rgba(52,211,153,0.18) 0%, rgba(16,185,129,0.08) 100%);
    border: 1.5px solid rgba(52,211,153,0.5);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-icon   { font-size: 2.8rem; line-height: 1.2; }
.result-title  { font-size: 1.5rem; font-weight: 700; margin: 0.4rem 0 0.2rem; }
.result-desc   { font-size: 0.9rem; color: #94a3b8; }
.pos-color     { color: #f87171; }
.neg-color     { color: #34d399; }

/* ── Input labels ── */
label[data-testid="stWidgetLabel"] p {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}

/* ── Number inputs ── */
input[type="number"] {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Predict button ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    color: #0a0f1e;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.04em;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

/* ── Dividers ── */
hr { border-color: rgba(56,189,248,0.1) !important; margin: 1.5rem 0 !important; }

/* ── Gauge track ── */
.gauge-label {
    font-size: 0.8rem;
    color: #64748b;
    text-align: center;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    missing = [f for f in ("diabetes_model.pkl", "scaler.pkl")
               if not os.path.exists(f)]
    if missing:
        return None, None, ", ".join(missing)
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler, None


model, scaler, load_err = load_model()



FIELDS = [
    # (key,                        label,                        unit,    min,   max,   step,  default)
    ("pregnancies",                "Pregnancies",                "count",   0,    17,    1,     0),
    ("glucose",                    "Glucose",                    "mg/dL",   0,   199,    1,   120),
    ("blood_pressure",             "Blood Pressure",             "mmHg",    0,   122,    1,    70),
    ("skin_thickness",             "Skin Thickness",             "mm",      0,    99,    1,    20),
    ("insulin",                    "Insulin",                    "μU/mL",   0,   846,    1,    80),
    ("bmi",                        "BMI",                        "kg/m²", 0.0,  67.1,  0.1,  25.0),
    ("diabetes_pedigree_function", "Diabetes Pedigree Function", "",      0.0,   2.5, 0.01, 0.47),
    ("age",                        "Age",                        "years",  21,    81,    1,    33),
]

HEALTHY_RANGES = {
    "pregnancies":                (0,    6,    "Normal: 0–6"),
    "glucose":                    (70,  140,   "Normal: 70–140 mg/dL"),
    "blood_pressure":             (60,   80,   "Optimal: 60–80 mmHg"),
    "skin_thickness":             (10,   30,   "Normal: 10–30 mm"),
    "insulin":                    (16,  166,   "Normal: 16–166 μU/mL"),
    "bmi":                        (18.5, 24.9, "Healthy: 18.5–24.9"),
    "diabetes_pedigree_function": (0.0,  1.0,  "Low risk: 0.0–1.0"),
    "age":                        (21,   40,   "Younger: lower risk"),
}



with st.sidebar:



    st.markdown("**How to use**")
    st.markdown("""
1. Enter all 8 values in the form
2. Click **Predict Risk**
3. Review the result and confidence gauge
""")

    if load_err:
        st.error(f"Missing: {load_err}\nRun `train_and_save_model.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main layout
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter patient clinical data to assess diabetes risk using machine learning.</div>', unsafe_allow_html=True)

if load_err:
    st.error(f"⚠️  Model files not found: **{load_err}**\n\n"
             "Run `python train_and_save_model.py` first, then restart the app.")
    st.stop()

# ── Input section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Data Input</div>', unsafe_allow_html=True)

col_groups = [FIELDS[:4], FIELDS[4:]]
cols_left, cols_right = st.columns(2, gap="large")

input_values = {}

for col_widget, group in zip([cols_left, cols_right], col_groups):
    with col_widget:
        for key, label, unit, mn, mx, step, default in group:
            display_label = f"{label}  ({unit})" if unit else label
            if isinstance(step, float):
                val = st.number_input(display_label, min_value=float(mn),
                                      max_value=float(mx), value=float(default),
                                      step=step, key=key)
            else:
                val = st.number_input(display_label, min_value=int(mn),
                                      max_value=int(mx), value=int(default),
                                      step=step, key=key)
            input_values[key] = val

st.markdown("")


_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    predict_clicked = st.button("  Predict Diabetes Risk", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)



if predict_clicked:
    feature_order = [k for k, *_ in FIELDS]
    arr    = np.array([input_values[k] for k in feature_order]).reshape(1, -1)
    scaled = scaler.transform(arr)
    pred   = model.predict(scaled)[0]


    try:
        proba             = model.predict_proba(scaled)[0]   # [prob_0, prob_1]
        diabetic_prob     = proba[1] * 100
        non_diabetic_prob = proba[0] * 100
        has_proba         = True
    except AttributeError:

        import math
        dec = float(model.decision_function(scaled)[0])

        diabetic_prob     = 100 / (1 + math.exp(-dec))
        non_diabetic_prob = 100 - diabetic_prob
        has_proba         = False   # flag to show note in UI

    res_col, gauge_col = st.columns([1.1, 1], gap="large")


    with res_col:
        if pred == 1:
            st.markdown(f"""
<div class="result-positive">
  <div class="result-icon">⚠️</div>
  <div class="result-title pos-color">High Diabetes Risk Detected</div>
  <div class="result-desc">
    The model predicts a <strong style="color:#f87171">{diabetic_prob:.1f}%</strong>
    probability of diabetes.<br>
    Please consult a healthcare professional for proper diagnosis.
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-negative">
  <div class="result-icon">✅</div>
  <div class="result-title neg-color">Low Diabetes Risk</div>
  <div class="result-desc">
    The model predicts a <strong style="color:#34d399">{non_diabetic_prob:.1f}%</strong>
    probability of no diabetes.<br>
    Maintain a healthy lifestyle and schedule regular check-ups.
  </div>
</div>""", unsafe_allow_html=True)


        st.markdown("")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-val">{diabetic_prob:.0f}%</div>
  <div class="metric-lbl">Diabetic Risk</div>
</div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-val">{non_diabetic_prob:.0f}%</div>
  <div class="metric-lbl">Non-Diabetic</div>
</div>""", unsafe_allow_html=True)
        with m3:
            conf = max(diabetic_prob, non_diabetic_prob)
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-val">{conf:.0f}%</div>
  <div class="metric-lbl">Confidence</div>
</div>""", unsafe_allow_html=True)

        if not has_proba:
            st.warning(
                "Your saved model was trained without probability=True. "
                "Probabilities shown are estimated via decision function (approximate). "
                "Delete the old .pkl files and re-run train_and_save_model.py for exact probabilities."
            )


    with gauge_col:
        st.markdown('<div class="section-label">Risk Probability Gauge</div>',
                    unsafe_allow_html=True)

        gauge_color = "#f87171" if pred == 1 else "#34d399"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=diabetic_prob,
            number={"suffix": "%", "font": {"size": 32, "color": "#f1f5f9",
                                             "family": "DM Mono"}},
            delta={"reference": 50,
                   "increasing": {"color": "#f87171"},
                   "decreasing": {"color": "#34d399"}},
            title={"text": "Diabetes Probability",
                   "font": {"size": 13, "color": "#64748b",
                             "family": "DM Sans"}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickcolor": "#334155",
                         "tickfont": {"color": "#64748b", "size": 10}},
                "bar":  {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "rgba(30,41,59,0.4)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  30], "color": "rgba(52,211,153,0.12)"},
                    {"range": [30, 60], "color": "rgba(251,191,36,0.10)"},
                    {"range": [60,100], "color": "rgba(248,113,113,0.12)"},
                ],
                "threshold": {
                    "line":  {"color": "#fbbf24", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=60, b=20, l=30, r=30),
            height=280,
            font={"family": "DM Sans", "color": "#94a3b8"},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="gauge-label">🟢 Low  ·  🟡 Moderate  ·  🔴 High Risk</div>',
                    unsafe_allow_html=True)


    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Feature Analysis — Out-of-Range Indicators</div>',
                unsafe_allow_html=True)

    flag_cols = st.columns(4)
    flags = []
    for i, (key, label, unit, *_) in enumerate(FIELDS):
        lo, hi, hint = HEALTHY_RANGES[key]
        v = input_values[key]
        in_range = lo <= v <= hi
        symbol   = "🟢" if in_range else "🔴"
        status   = "Normal" if in_range else "Abnormal"
        flags.append((symbol, label, f"{v} {unit}".strip(), status, hint))

    for i, (sym, lbl, val, status, hint) in enumerate(flags):
        with flag_cols[i % 4]:
            st.markdown(f"""
<div style="background:rgba(30,41,59,0.5);border:1px solid rgba(56,189,248,0.1);
border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.8rem;">
  <div style="font-size:1.3rem;margin-bottom:0.2rem">{sym}</div>
  <div style="font-size:0.8rem;font-weight:600;color:#cbd5e1">{lbl}</div>
  <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#38bdf8">{val}</div>
  <div style="font-size:0.7rem;color:#64748b;margin-top:0.2rem">{hint}</div>
</div>""", unsafe_allow_html=True)

else:

    st.markdown("""
<div style="text-align:center;padding:3rem 1rem;color:#334155;">
  <div style="font-size:3.5rem;margin-bottom:1rem">🩺</div>
  <div style="font-size:1.1rem;color:#475569;font-weight:500">
    Fill in the patient data above and click <strong style="color:#38bdf8">Predict Diabetes Risk</strong>
  </div>
  <div style="font-size:0.85rem;color:#334155;margin-top:0.5rem">
    Results include probability score, confidence gauge, and feature analysis.
  </div>
</div>
""", unsafe_allow_html=True)