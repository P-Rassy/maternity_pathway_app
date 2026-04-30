import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import json
import tempfile
import os

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Maternity Risk Predictor",
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STYLES
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }
    .stApp { background-color: #F7F4EF; }
    section[data-testid="stSidebar"] { background-color: #1C2B3A; }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] label {
        color: #A8BFD0 !important;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .risk-card {
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .risk-low      { background: linear-gradient(135deg,#E8F5E9,#F1F8E9); border-left:5px solid #4CAF50; }
    .risk-moderate { background: linear-gradient(135deg,#FFF8E1,#FFF3CD); border-left:5px solid #FF9800; }
    .risk-high     { background: linear-gradient(135deg,#FFEBEE,#FCE4EC); border-left:5px solid #F44336; }
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-box .value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'DM Serif Display', serif;
        color: #1C2B3A;
    }
    .metric-box .label {
        font-size: 0.72rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.05rem;
        color: #1C2B3A;
        border-bottom: 2px solid #E5DDD0;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        margin-top: 1.8rem;
    }
    .stButton > button {
        background-color: #1C2B3A;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
    }
    .stButton > button:hover { background-color: #2C3E50; }
    .warning-box {
        background: #FFF3CD; border: 1px solid #FFD700;
        border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 0.85rem; color: #856404;
    }
    .info-box {
        background: #E3F2FD; border: 1px solid #90CAF9;
        border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 0.85rem; color: #1565C0; margin-top: 1rem;
    }
    .tier-table { width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:0.5rem; }
    .tier-table th { background:#1C2B3A; color:white; padding:0.5rem 0.8rem; text-align:left; }
    .tier-table td { padding:0.5rem 0.8rem; border-bottom:1px solid #E5DDD0; }
    .tier-table tr:nth-child(even) td { background:#F7F4EF; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# URLs — replace with your actual GitHub raw URLs
# =============================================================================

MODEL_URL     = "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/calibrated_model.joblib"
ARTEFACTS_URL = "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/model_artefacts.json"

# =============================================================================
# LOADERS
# =============================================================================

@st.cache_resource
def load_model_from_url(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        f.write(r.content)
        tmp = f.name
    model = joblib.load(tmp)
    os.unlink(tmp)
    return model

@st.cache_data
def load_artefacts_from_url(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

# =============================================================================
# FEATURE DEFINITIONS
# All 16 features from model_artefacts.json, split into:
#   - Direct inputs  (entered by clinician as-is)
#   - Raw inputs     (used to compute derived features)
# =============================================================================

# Direct inputs
DIRECT_FEATURES = {
    # Demographics
    "age": {
        "label": "Maternal Age", "type": "int",
        "min": 15, "max": 50, "default": 28, "unit": "years",
        "section": "Demographics",
    },
    # Anthropometric
    "bmi_first_visit": {
        "label": "BMI (1st Visit)", "type": "float",
        "min": 15.0, "max": 55.0, "default": 23.0, "unit": "kg/m²",
        "section": "Anthropometric",
    },
    # Obstetric history
    "prev_pregnancies__first_visit": {
        "label": "Previous Pregnancies", "type": "int",
        "min": 0, "max": 12, "default": 0, "unit": "",
        "section": "Obstetric History",
    },
    "prev_deliveries__first_visit": {
        "label": "Previous Deliveries", "type": "int",
        "min": 0, "max": 12, "default": 0, "unit": "",
        "section": "Obstetric History",
    },
    "prev_c_sections__first_visit": {
        "label": "Previous C-Sections", "type": "int",
        "min": 0, "max": 12, "default": 0, "unit": "",
        "section": "Obstetric History",
    },
    "multiple_gestation__third_trimester": {
        "label": "Multiple Gestation", "type": "bool",
        "default": 0, "unit": "",
        "section": "Obstetric History",
    },
    "num_fetuses__third_trimester": {
        "label": "Number of Fetuses", "type": "int",
        "min": 1, "max": 5, "default": 1, "unit": "",
        "section": "Obstetric History",
    },
    # Clinical flags
    "has_comorbidity": {
        "label": "Has Comorbidity", "type": "bool",
        "default": 0, "unit": "",
        "section": "Clinical Flags",
    },
    "has_anomaly": {
        "label": "Has Anomaly", "type": "bool",
        "default": 0, "unit": "",
        "section": "Clinical Flags",
    },
    "any_domain_worsened": {
        "label": "Any PROMs Domain Worsened (1st → 3rd)", "type": "bool",
        "default": 0, "unit": "",
        "section": "Clinical Flags",
    },
    "poor_care_perceived": {
        "label": "Poor Perceived Care", "type": "bool",
        "default": 0, "unit": "",
        "section": "Clinical Flags",
    },
    "poor_health_1st": {
        "label": "Poor Health at 1st Visit", "type": "bool",
        "default": 0, "unit": "",
        "section": "Clinical Flags",
    },
}

# Raw inputs — used only to compute derived features
RAW_INPUTS = {
    "phq2_total__first_visit": {
        "label": "PHQ-2 Score (1st Visit)", "type": "float",
        "min": 0.0, "max": 6.0, "default": 0.0, "unit": "",
        "section": "PRO Scores (for derived features)",
    },
    "weight_gain": {
        "label": "Weight Gain (1st → 3rd Trimester)", "type": "float",
        "min": -5.0, "max": 40.0, "default": 12.0, "unit": "kg",
        "section": "PRO Scores (for derived features)",
    },
    "eq5d_3l_healthtoday__first_visit": {
        "label": "Health Today VAS (1st Visit)", "type": "float",
        "min": 0.0, "max": 100.0, "default": 75.0, "unit": "/100",
        "section": "PRO Scores (for derived features)",
    },
    "eq5d_3l_healthtoday__third_trimester": {
        "label": "Health Today VAS (3rd Trimester)", "type": "float",
        "min": 0.0, "max": 100.0, "default": 70.0, "unit": "/100",
        "section": "PRO Scores (for derived features)",
    },
    "wexner_total__first_visit": {
        "label": "Wexner Score (1st Visit)", "type": "float",
        "min": 0.0, "max": 20.0, "default": 0.0, "unit": "",
        "section": "PRO Scores (for derived features)",
    },
    "wexner_total__third_trimester": {
        "label": "Wexner Score (3rd Trimester)", "type": "float",
        "min": 0.0, "max": 20.0, "default": 0.0, "unit": "",
        "section": "PRO Scores (for derived features)",
    },
}

# =============================================================================
# HELPER
# =============================================================================

def render_input(feat, cfg, key):
    label = cfg["label"]
    if cfg["type"] == "bool":
        return st.selectbox(label, options=[0, 1],
                            format_func=lambda x: "Yes" if x else "No",
                            index=int(cfg["default"]), key=key)
    elif cfg["type"] == "int":
        return st.number_input(label, min_value=int(cfg["min"]), max_value=int(cfg["max"]),
                               value=int(cfg["default"]), step=1, key=key)
    else:
        return st.number_input(label, min_value=float(cfg["min"]), max_value=float(cfg["max"]),
                               value=float(cfg["default"]), step=0.1, key=key)

def compute_derived(inputs):
    age  = inputs.get("age", np.nan)
    phq2 = inputs.get("phq2_total__first_visit", np.nan)
    bmi  = inputs.get("bmi_first_visit", np.nan)
    wg   = inputs.get("weight_gain", np.nan)
    ht1  = inputs.get("eq5d_3l_healthtoday__first_visit", np.nan)
    ht3  = inputs.get("eq5d_3l_healthtoday__third_trimester", np.nan)
    wx1  = inputs.get("wexner_total__first_visit", np.nan)
    wx3  = inputs.get("wexner_total__third_trimester", np.nan)
    return {
        "age_x_phq2":         age * phq2,
        "bmi_x_weight_gain":  bmi * wg,
        "healthtoday_change":  ht3 - ht1,
        "wexner_change":       wx3 - wx1,
    }

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 📊 Model Info")
    st.markdown("---")
    try:
        _art = load_artefacts_from_url(ARTEFACTS_URL)
        st.markdown(f"**Model:** {_art['model_name'].replace('_',' ').title()}")
        st.markdown(f"**ROC-AUC:** {_art['metrics']['roc_auc']:.3f}")
        st.markdown(f"**Recall:** {_art['metrics']['recall']:.3f}")
        st.markdown(f"**Precision:** {_art['metrics']['precision']:.3f}")
        st.markdown(f"**Features:** {len(_art['features'])}")
        st.markdown("---")
        st.markdown("**Thresholds**")
        st.markdown(f"🟡 Screening : `{_art['thresholds']['screening']:.3f}`")
        st.markdown(f"🔴 High Risk : `{_art['thresholds']['high_risk']:.3f}`")
    except Exception:
        st.warning("Model info unavailable")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#A8BFD0;'>For research use only.<br>Not a substitute for clinical judgment.</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# HEADER
# =============================================================================

st.markdown("# 🤱 Maternity Complication Risk Predictor")
st.markdown(
    "<p style='color:#6B7280;font-size:0.93rem;'>Enter patient data to generate a complication risk estimate. "
    "The model uses two clinical thresholds: <b>Screening</b> (high sensitivity, flags 34% of patients) "
    "and <b>High Risk</b> (high specificity, flags 5% of patients).</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# =============================================================================
# INPUT FORM
# =============================================================================

all_inputs = {}

# Direct features grouped by section
direct_sections = {}
for feat, cfg in DIRECT_FEATURES.items():
    direct_sections.setdefault(cfg["section"], {})[feat] = cfg

for section_name, feats in direct_sections.items():
    st.markdown(f"<div class='section-header'>{section_name}</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (feat, cfg) in enumerate(feats.items()):
        with cols[idx % 3]:
            all_inputs[feat] = render_input(feat, cfg, key=feat)

# Raw inputs for derived features
raw_sections = {}
for feat, cfg in RAW_INPUTS.items():
    raw_sections.setdefault(cfg["section"], {})[feat] = cfg

for section_name, feats in raw_sections.items():
    st.markdown(f"<div class='section-header'>{section_name}</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (feat, cfg) in enumerate(feats.items()):
        with cols[idx % 3]:
            all_inputs[feat] = render_input(feat, cfg, key=feat)

# =============================================================================
# PREDICT BUTTON
# =============================================================================

st.markdown("---")
btn_col, _ = st.columns([1, 2])
with btn_col:
    predict_clicked = st.button("🔍 Generate Risk Prediction")

if predict_clicked:
    with st.spinner("Loading model and computing risk..."):
        try:
            model     = load_model_from_url(MODEL_URL)
            artefacts = load_artefacts_from_url(ARTEFACTS_URL)
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()

        # Build input row
        derived  = compute_derived(all_inputs)
        combined = {**all_inputs, **derived}
        features = artefacts["features"]
        input_df = pd.DataFrame([{f: combined.get(f, np.nan) for f in features}])

        thr_s = artefacts["thresholds"]["screening"]
        thr_h = artefacts["thresholds"]["high_risk"]

        try:
            prob = float(model.predict_proba(input_df)[0, 1])
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        # Risk tier
        if prob >= thr_h:
            risk_level, risk_class, risk_icon = "High Risk",       "risk-high",     "🔴"
            risk_msg = "Patient exceeds the high-risk threshold. Immediate clinical review is recommended."
        elif prob >= thr_s:
            risk_level, risk_class, risk_icon = "Screening Flag",  "risk-moderate", "🟡"
            risk_msg = "Patient exceeds the screening threshold. Closer follow-up and additional assessment are advised."
        else:
            risk_level, risk_class, risk_icon = "Low Risk",        "risk-low",      "🟢"
            risk_msg = "Patient is below the screening threshold. Continue standard monitoring."

        # Risk card
        st.markdown("### 📊 Prediction Results")
        st.markdown(f"""
        <div class='risk-card {risk_class}'>
            <div style='font-size:1.7rem;font-weight:700;font-family:DM Serif Display,serif;'>
                {risk_icon} {risk_level}
            </div>
            <div style='font-size:2.8rem;font-weight:800;color:#1C2B3A;margin:0.4rem 0;'>
                {prob:.1%}
            </div>
            <div style='font-size:0.9rem;color:#4B5563;'>{risk_msg}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metric boxes
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{prob:.3f}",    "Predicted Probability"),
            (c2, risk_level,       "Risk Tier"),
            (c3, f"{thr_s:.3f}",  "Screening Threshold"),
            (c4, f"{thr_h:.3f}",  "High-Risk Threshold"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-box'>
                    <div class='value'>{val}</div>
                    <div class='label'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        # Threshold performance table
        tier_m = artefacts.get("tier_metrics", {})
        if tier_m:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Threshold Performance Reference")
            st.markdown(f"""
            <table class='tier-table'>
                <thead>
                    <tr><th>Tier</th><th>Threshold</th><th>Recall</th><th>Precision</th><th>F2</th><th>Alert Rate</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>🟡 Screening</td>
                        <td>{thr_s:.3f}</td>
                        <td>{tier_m['screening']['recall']:.3f}</td>
                        <td>{tier_m['screening']['precision']:.3f}</td>
                        <td>{tier_m['screening']['f2']:.3f}</td>
                        <td>{tier_m['screening']['alert_rate']:.1%}</td>
                    </tr>
                    <tr>
                        <td>🔴 High Risk</td>
                        <td>{thr_h:.3f}</td>
                        <td>{tier_m['high_risk']['recall']:.3f}</td>
                        <td>{tier_m['high_risk']['precision']:.3f}</td>
                        <td>{tier_m['high_risk']['f2']:.3f}</td>
                        <td>{tier_m['high_risk']['alert_rate']:.1%}</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-box'>
            ℹ️ This prediction is generated by a machine learning model trained on clinical data.
            It is intended to support — not replace — clinical judgment. Always interpret results
            in the context of the full clinical picture.
        </div>
        """, unsafe_allow_html=True)
