import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }

    .stApp {
        background-color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #1C2B3A;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

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

    .risk-low {
        background: linear-gradient(135deg,#E8F5E9,#F1F8E9);
        border-left: 5px solid #4CAF50;
    }

    .risk-moderate {
        background: linear-gradient(135deg,#FFF8E1,#FFF3CD);
        border-left: 5px solid #FF9800;
    }

    .risk-high {
        background: linear-gradient(135deg,#FFEBEE,#FCE4EC);
        border-left: 5px solid #F44336;
    }

    .risk-title {
        font-size: 1.7rem;
        font-weight: 700;
        font-family: 'DM Serif Display', serif;
        color: #1C2B3A;
    }

    .risk-probability {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1C2B3A;
        margin: 0.4rem 0;
    }

    .risk-message {
        font-size: 0.9rem;
        color: #4B5563;
    }

    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.05);
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

    .stButton > button:hover {
        background-color: #2C3E50;
    }

    .info-box {
        background: #E3F2FD;
        border: 1px solid #90CAF9;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #1565C0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOCAL FILE PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "calibrated_model.joblib")
ARTEFACTS_PATH = os.path.join(BASE_DIR, "model_artefacts.json")

# =============================================================================
# LOADERS
# =============================================================================

@st.cache_resource
def load_model_local(path):
    return joblib.load(path)

@st.cache_data
def load_artefacts_local(path):
    with open(path, "r") as f:
        return json.load(f)

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

DIRECT_FEATURES = {
    "age": {
        "label": "Maternal Age",
        "type": "int",
        "min": 15,
        "max": 50,
        "default": 28,
    },
    "bmi_first_visit": {
        "label": "BMI (1st Visit)",
        "type": "float",
        "min": 15.0,
        "max": 55.0,
        "default": 23.0,
    },
    "prev_pregnancies__first_visit": {
        "label": "Previous Pregnancies",
        "type": "int",
        "min": 0,
        "max": 12,
        "default": 0,
    },
    "prev_deliveries__first_visit": {
        "label": "Previous Deliveries",
        "type": "int",
        "min": 0,
        "max": 12,
        "default": 0,
    },
    "prev_c_sections__first_visit": {
        "label": "Previous C-Sections",
        "type": "int",
        "min": 0,
        "max": 12,
        "default": 0,
    },
    "multiple_gestation__third_trimester": {
        "label": "Multiple Gestation",
        "type": "bool",
        "default": 0,
    },
    "num_fetuses__third_trimester": {
        "label": "Number of Fetuses",
        "type": "int",
        "min": 1,
        "max": 5,
        "default": 1,
    },
    "has_comorbidity": {
        "label": "Has Comorbidity",
        "type": "bool",
        "default": 0,
    },
    "has_anomaly": {
        "label": "Has Anomaly",
        "type": "bool",
        "default": 0,
    },
    "any_domain_worsened": {
        "label": "Any PROMs Domain Worsened (1st → 3rd)",
        "type": "bool",
        "default": 0,
    },
    "poor_care_perceived": {
        "label": "Poor Perceived Care",
        "type": "bool",
        "default": 0,
    },
    "poor_health_1st": {
        "label": "Poor Health at 1st Visit",
        "type": "bool",
        "default": 0,
    },
}

RAW_INPUTS = {
    "phq2_total__first_visit": {
        "label": "PHQ-2 Score (1st Visit)",
        "type": "float",
        "min": 0.0,
        "max": 6.0,
        "default": 0.0,
    },
    "weight_gain": {
        "label": "Weight Gain (1st → 3rd Trimester)",
        "type": "float",
        "min": -5.0,
        "max": 40.0,
        "default": 12.0,
    },
    "eq5d_3l_healthtoday__first_visit": {
        "label": "Health Today VAS (1st Visit)",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
        "default": 75.0,
    },
    "eq5d_3l_healthtoday__third_trimester": {
        "label": "Health Today VAS (3rd Trimester)",
        "type": "float",
        "min": 0.0,
        "max": 100.0,
        "default": 70.0,
    },
    "wexner_total__first_visit": {
        "label": "Wexner Score (1st Visit)",
        "type": "float",
        "min": 0.0,
        "max": 20.0,
        "default": 0.0,
    },
    "wexner_total__third_trimester": {
        "label": "Wexner Score (3rd Trimester)",
        "type": "float",
        "min": 0.0,
        "max": 20.0,
        "default": 0.0,
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def render_input(feat, cfg, key):
    label = cfg["label"]

    if cfg["type"] == "bool":
        return st.selectbox(
            label,
            options=[0, 1],
            format_func=lambda x: "Yes" if x else "No",
            index=int(cfg["default"]),
            key=key
        )

    elif cfg["type"] == "int":
        return st.number_input(
            label,
            min_value=int(cfg["min"]),
            max_value=int(cfg["max"]),
            value=int(cfg["default"]),
            step=1,
            key=key
        )

    else:
        return st.number_input(
            label,
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=0.1,
            key=key
        )


def compute_derived(inputs):
    age = inputs.get("age", np.nan)
    phq2 = inputs.get("phq2_total__first_visit", np.nan)
    bmi = inputs.get("bmi_first_visit", np.nan)
    wg = inputs.get("weight_gain", np.nan)
    ht1 = inputs.get("eq5d_3l_healthtoday__first_visit", np.nan)
    ht3 = inputs.get("eq5d_3l_healthtoday__third_trimester", np.nan)
    wx1 = inputs.get("wexner_total__first_visit", np.nan)
    wx3 = inputs.get("wexner_total__third_trimester", np.nan)

    return {
        "age_x_phq2": age * phq2,
        "bmi_x_weight_gain": bmi * wg,
        "healthtoday_change": ht3 - ht1,
        "wexner_change": wx3 - wx1,
    }

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        "<div style='font-size:0.72rem;color:#A8BFD0;'>For research use only.<br>Not a substitute for clinical judgment.</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# HEADER
# =============================================================================

st.markdown("# 🤱 Maternity Complication Risk Predictor")

st.markdown(
    "<p style='color:#6B7280;font-size:0.93rem;'>Enter patient data to generate a complication risk estimate.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# =============================================================================
# INPUT FORM
# =============================================================================

all_inputs = {}

all_features = {**DIRECT_FEATURES, **RAW_INPUTS}

feature_items = list(all_features.items())

cols = st.columns(3)

for idx, (feat, cfg) in enumerate(feature_items):
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
            model = load_model_local(MODEL_PATH)
            artefacts = load_artefacts_local(ARTEFACTS_PATH)

        except Exception as e:
            st.error(f"❌ Failed to load local files: {e}")
            st.stop()

        derived = compute_derived(all_inputs)
        combined = {**all_inputs, **derived}

        features = artefacts["features"]

        input_df = pd.DataFrame([
            {f: combined.get(f, np.nan) for f in features}
        ])

        thr_s = artefacts["thresholds"]["screening"]
        thr_h = artefacts["thresholds"]["high_risk"]

        try:
            prob = float(model.predict_proba(input_df)[0, 1])

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        if prob >= thr_h:
            risk_level = "High Risk"
            risk_class = "risk-high"
            risk_icon = "🔴"
            risk_msg = "High risk — immediate clinical attention recommended."

        elif prob >= thr_s:
            risk_level = "Moderate Risk"
            risk_class = "risk-moderate"
            risk_icon = "🟡"
            risk_msg = "Moderate risk — closer monitoring advised."

        else:
            risk_level = "Low Risk"
            risk_class = "risk-low"
            risk_icon = "🟢"
            risk_msg = "Low risk — continue routine care."

        st.markdown(
            f"""
            <div class="{risk_class} risk-card">
                <div class="risk-title">{risk_icon} {risk_level}</div>
                <div class="risk-probability">{prob:.1%}</div>
                <div class="risk-message">{risk_msg}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        metrics = [
            (c1, f"{prob:.3f}", "Predicted Probability"),
            (c2, risk_level, "Risk Level"),
            (c3, f"{thr_s:.3f}", "Moderate Threshold"),
            (c4, f"{thr_h:.3f}", "High Threshold"),
        ]

        for col, val, lbl in metrics:
            with col:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="value">{val}</div>
                        <div class="label">{lbl}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
            """
            <div class="info-box">
                ℹ️ This prediction is generated by a machine learning model trained on clinical data.
                It is intended to support — not replace — clinical judgment. Always interpret results
                in the context of the full clinical picture.
            </div>
            """,
            unsafe_allow_html=True
        )
