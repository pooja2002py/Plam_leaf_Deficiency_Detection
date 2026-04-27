import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Leaf Detection 🌿",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# BACKGROUND IMAGE FUNCTION (FIXED)
# -------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_file):
    try:
        img_base64 = get_base64_image(image_file)

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.60);
            z-index: 0;
        }}

        .main .block-container {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.warning("⚠️ Background image not found. Using fallback color.")
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(120deg, #141e30, #243b55);
        }
        </style>
        """, unsafe_allow_html=True)

# ✅ IMPORTANT CHANGE HERE 👇
set_background("Agriculture.jpg")   # image same folder me rakho

# -------------------------------
# UI DESIGN (same as yours)
# -------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: #fff;
}
.sub-title {
    text-align: center;
    color: #e0e0e0;
    font-size: 18px;
}
.main-box {
    background: rgba(255,255,255,0.10);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(8px);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='main-title'>🌿 AI-Based Palm Leaf Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI Powered Crop Health Analysis using YOLOv8</div>", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------
# IMAGE INPUT
# -------------------------------
st.markdown("## 📥 Upload Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("🚀 Detect"):
        img_np = np.array(image)
        img_np = img_np[:, :, ::-1]

        results = model(img_np)
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Detected Image")

        boxes = results[0].boxes
        if boxes and len(boxes) > 0:
            best = max(boxes, key=lambda x: float(x.conf[0]))
            label = model.names[int(best.cls[0])]
            conf = float(best.conf[0])

            st.success(f"Detected: {label} ({conf:.2f})")
        else:
            st.warning("No detection found ❌")

else:
    st.info("Upload image first")
