import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Brain Tumor Diagnosis",
    page_icon="🧠",
    layout="centered"
)

# ================== CUSTOM CSS (Hospital Style) ==================
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 20px;
}
h1 {
    color: #0b5394;
    font-weight: 700;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
}
.positive {
    background-color: #FB4549;
    border-left: 6px solid #d9534f;
}
.negative {
    background-color: #009900;
    border-left: 6px solid #2ecc71;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_brain_model():
    return load_model("Brain_Tumor_dataset.h5")

model = load_brain_model()

# ================== HEADER ==================
st.markdown("<h1 style='text-align:center;'>🧠 Brain Tumor Diagnosis System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-assisted MRI image analysis</p>", unsafe_allow_html=True)
st.markdown("---")

# ================== IMAGE INPUT ==================
tab1, tab2 = st.tabs(["📁 Upload MRI Image", "📷 Capture Using Camera"])

uploaded_image = None

with tab1:
    uploaded_image = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

with tab2:
    uploaded_image = st.camera_input("Take MRI Photo")

# ================== PREDICTION ==================
if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]

    tumor_prob = float(prediction)
    no_tumor_prob = 1 - tumor_prob

    st.markdown("## 🔬 Diagnostic Result")

    # ================== RESULT BOX ==================
    if tumor_prob >= 0.5:
        st.markdown(
            f"""
            <div class="result-box positive">
            <h3>⚠️ Brain Tumor Detected</h3>
            <p>Immediate medical consultation is recommended.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-box negative">
            <h3>✅ No Brain Tumor Detected</h3>
            <p>No abnormal tumor patterns found.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ================== PROBABILITY BAR ==================
    st.markdown("### 📊 Prediction Confidence")

    st.write("**Tumor Probability**")
    st.progress(tumor_prob)

    st.write("**No Tumor Probability**")
    st.progress(no_tumor_prob)

    # ================== NUMERIC VALUES ==================
    col1, col2 = st.columns(2)
    col1.metric("Tumor Probability", f"{tumor_prob*100:.2f}%")
    col2.metric("No Tumor Probability", f"{no_tumor_prob*100:.2f}%")

# ================== FOOTER ==================
st.markdown("---")
st.markdown(
    "<div class='footer'>⚕️ AI-assisted system | Not a substitute for professional diagnosis</div>",
    unsafe_allow_html=True
)
