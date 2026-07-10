import streamlit as st
import numpy as np
from PIL import Image
import joblib

# --------------------------------------
# Page Config
# --------------------------------------

st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------
# Custom CSS
# --------------------------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#00D4FF;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#DDDDDD;
    margin-bottom:25px;
}

.card{
    background:#1C1C1C;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,212,255,0.3);
}

.result{
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------
# Load Model
# --------------------------------------

model = joblib.load("gender_classification_model.pkl")

IMG_SIZE = 64

# --------------------------------------
# Header
# --------------------------------------

st.markdown('<div class="title">👨👩 Face Gender Classification</div>', unsafe_allow_html=True)

st.markdown(
'<div class="subtitle">Upload a face image and let AI predict whether it belongs to a Male or Female.</div>',
unsafe_allow_html=True
)

# --------------------------------------
# Upload
# --------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns([1,1])

    with col1:
        st.image(image,use_container_width=True)

    resized=image.resize((IMG_SIZE,IMG_SIZE))
    resized=np.array(resized).flatten()

    prediction=model.predict([resized])[0]
    probability=model.predict_proba([resized])[0]

    male=probability[0]*100
    female=probability[1]*100

    with col2:

        st.markdown('<div class="card">',unsafe_allow_html=True)

        if prediction==0:

            st.success("Prediction: MALE 👨")

        else:

            st.success("Prediction: FEMALE 👩")

        st.markdown("### Confidence")

        st.write(f"👨 Male : **{male:.2f}%**")
        st.progress(float(male)/100)

        st.write(f"👩 Female : **{female:.2f}%**")
        st.progress(float(female)/100)

        st.markdown("</div>",unsafe_allow_html=True)

st.markdown("---")

st.caption("Built using Streamlit • Scikit-learn • NumPy • Pillow")
