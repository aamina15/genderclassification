
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
 
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Face Gender Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ============================================================
# CUSTOM CSS — PROFESSIONAL DASHBOARD THEME
# ============================================================
st.markdown("""
<style>
 
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
 
* { font-family: 'Inter', sans-serif; }
 
.stApp {
    background: radial-gradient(circle at top left, #171B26 0%, #0B0D13 55%, #0B0D13 100%);
    color: #E6E8EC;
}
 
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14161F 0%, #0E1016 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
 
h1, h2, h3, h4, h5, h6 { color: #F5F6FA !important; font-weight: 700 !important; }
p, label, span { color: #C7CBD4; }
 
/* Hero header */
.hero {
    background: linear-gradient(135deg, rgba(0,212,255,0.14) 0%, rgba(255,45,149,0.08) 100%);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
    text-align: center;
}
 
.hero h1 {
    font-size: 2.2rem !important;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #00D4FF 0%, #FF2D95 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
 
.hero p {
    font-size: 1rem;
    color: #B8BCC8 !important;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.6;
}
 
/* Cards */
.card {
    background: linear-gradient(180deg, #1A1D27 0%, #15171F 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.6rem 1.7rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
 
/* Result banner */
.result-banner {
    text-align: center;
    border-radius: 14px;
    padding: 1.1rem 1rem;
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 1.2rem;
    letter-spacing: 0.3px;
}
 
.result-male {
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(0,212,255,0.05));
    border: 1px solid rgba(0,212,255,0.45);
    color: #00D4FF;
}
 
.result-female {
    background: linear-gradient(135deg, rgba(255,45,149,0.18), rgba(255,45,149,0.05));
    border: 1px solid rgba(255,45,149,0.45);
    color: #FF2D95;
}
 
/* Confidence rows */
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
    color: #E6E8EC !important;
}
 
.stProgress > div > div {
    border-radius: 8px;
}
 
/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: #1A1D27;
    border: 1.5px dashed rgba(255,255,255,0.18);
    border-radius: 12px;
}
 
/* Alerts / info */
div[data-testid="stAlert"] { border-radius: 12px; }
 
hr { border-color: rgba(255,255,255,0.08) !important; }
 
.footer-box { text-align: center; padding: 1.4rem 0 0.4rem 0; }
.footer-box p { color: #7A7F8C !important; font-size: 0.85rem; }
 
.badge-pill {
    display: inline-block;
    width: 100%;
    background: #1A1D27;
    border: 1px solid rgba(255,255,255,0.08);
    color: #C7CBD4 !important;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.5rem 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
}
 
</style>
""", unsafe_allow_html=True)
 
# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "gender_classification_model.pkl"
IMG_SIZE = 64
 
# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🧠 Gender Classifier")
st.sidebar.caption("AI-powered facial gender prediction")
st.sidebar.markdown("---")
 
st.sidebar.markdown("**Built with**")
st.sidebar.markdown('<div class="badge-pill">🧬 Scikit-learn</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="badge-pill">🖼️ Pillow + NumPy</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="badge-pill">🚀 Streamlit</div>', unsafe_allow_html=True)
 
st.sidebar.markdown("---")
st.sidebar.markdown("**Settings**")
 
# Manual override in case a differently-trained model has the opposite
# class order to the one detected below. Off by default.
swap_labels = st.sidebar.checkbox(
    "Swap Male / Female labels",
    value=False,
    help="Enable only if predictions still look reversed after this fix."
)
 
show_debug = st.sidebar.checkbox("Show model debug info", value=False)
 
# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)
 
model = load_model(MODEL_PATH)
 
if model is None:
    st.error(
        f"⚠️ Model file `{MODEL_PATH}` was not found. "
        "Place it in the same folder as this app and refresh."
    )
    st.stop()
 
# ------------------------------------------------------------
# THE BUG FIX
# ------------------------------------------------------------
# scikit-learn stores `model.classes_` sorted in the order it saw
# labels during training. If training used string labels ("female",
# "male") or 0/1 encoded alphabetically, class index 0 is usually
# "Female" and index 1 is "Male" — the OPPOSITE of what the original
# app assumed (`prediction == 0 -> Male`), which is what caused male
# photos to be reported as female.
#
# Instead of hardcoding index -> label, we read the model's actual
# `classes_` attribute and build the mapping dynamically so it always
# matches how the model was really trained.
# ------------------------------------------------------------
raw_classes = list(model.classes_)
 
 
def resolve_label(raw_value):
    """Turn a raw class value (0/1 or 'male'/'female') into a clean label."""
    s = str(raw_value).strip().lower()
    if s in ("1", "male", "m"):
        return "Male"
    if s in ("0", "female", "f"):
        return "Female"
    # Fallback: alphabetical convention (female < male)
    return "Female" if raw_value == sorted(raw_classes)[0] else "Male"
 
 
class_to_label = {cls: resolve_label(cls) for cls in raw_classes}
 
if swap_labels:
    class_to_label = {
        cls: ("Female" if lbl == "Male" else "Male")
        for cls, lbl in class_to_label.items()
    }
 
# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>👨‍🦱👩 Face Gender Classifier</h1>
    <p>Upload a face photo and the model will predict whether it belongs to a
    male or female, along with a confidence breakdown for both classes.</p>
</div>
""", unsafe_allow_html=True)
 
if show_debug:
    with st.expander("🔍 Model debug info", expanded=True):
        st.write("**Raw `model.classes_`:**", raw_classes)
        st.write("**Resolved label mapping:**", class_to_label)
        st.caption(
            "If a photo you know is male keeps coming back as Female (or vice "
            "versa) even with this fix, enable **Swap Male / Female labels** "
            "in the sidebar — it flips the mapping instantly without retraining."
        )
 
# ============================================================
# UPLOAD + PREDICTION
# ============================================================
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
 
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
 
    col1, col2 = st.columns([1, 1], gap="large")
 
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded image")
        st.markdown("</div>", unsafe_allow_html=True)
 
    # preprocessing — must match how the model was trained
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    flattened = np.array(resized).flatten().reshape(1, -1)
 
    raw_prediction = model.predict(flattened)[0]
    probabilities = model.predict_proba(flattened)[0]
 
    # map each class's probability to its resolved label
    label_prob = {
        class_to_label[cls]: prob
        for cls, prob in zip(model.classes_, probabilities)
    }
    male_prob = label_prob.get("Male", 0.0) * 100
    female_prob = label_prob.get("Female", 0.0) * 100
    predicted_label = class_to_label[raw_prediction]
 
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
 
        if predicted_label == "Male":
            st.markdown(
                '<div class="result-banner result-male">Prediction: MALE 👨</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-banner result-female">Prediction: FEMALE 👩</div>',
                unsafe_allow_html=True
            )
 
        st.markdown("#### Confidence Breakdown")
 
        st.markdown(
            f'<div class="conf-label"><span>👨 Male</span><span>{male_prob:.2f}%</span></div>',
            unsafe_allow_html=True
        )
        st.progress(float(male_prob) / 100)
 
        st.markdown(
            f'<div class="conf-label"><span>👩 Female</span><span>{female_prob:.2f}%</span></div>',
            unsafe_allow_html=True
        )
        st.progress(float(female_prob) / 100)
 
        st.markdown("</div>", unsafe_allow_html=True)
 
else:
    st.info("👆 Upload a JPG or PNG face photo to get a prediction.")
 
# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="footer-box">
    <p>Built using Streamlit • Scikit-learn • NumPy • Pillow</p>
</div>
""", unsafe_allow_html=True)
