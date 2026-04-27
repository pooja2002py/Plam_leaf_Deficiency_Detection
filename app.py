import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Leaf Detection 🌿",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# BACKGROUND IMAGE
# -------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_file):
    try:
        img = get_base64_image(image_file)
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """, unsafe_allow_html=True)
    except:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(120deg, #141e30, #243b55);
        }
        </style>
        """, unsafe_allow_html=True)

set_background("Agriculture.jpg")

# -------------------------------
# TITLE
# -------------------------------
st.title("🌿 AI-Based Leaf Detection System")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------------------
# DEFICIENCY DATA
# -------------------------------
data = {
    "nitrogen": {
        "English": {
            "name": "Nitrogen Deficiency",
            "symptoms": ["Yellow leaves", "Slow growth"],
            "solution": ["Apply urea fertilizer", "Use compost"],
            "advice": ["Check soil regularly"]
        },
        "Telugu": {
            "name": "నైట్రోజన్ లోపం",
            "symptoms": ["ఆకులు పసుపు రంగులో మారడం"],
            "solution": ["యూరియా వేయండి"],
            "advice": ["మట్టి పరీక్ష చేయండి"]
        },
        "Marathi": {
            "name": "नायट्रोजन कमतरता",
            "symptoms": ["पाने पिवळी होतात"],
            "solution": ["युरिया वापरा"],
            "advice": ["माती तपासा"]
        }
    },
    "healthy": {
        "English": {
            "name": "Healthy Leaf",
            "symptoms": ["Green leaves"],
            "solution": ["No action needed"],
            "advice": ["Maintain care"]
        },
        "Telugu": {
            "name": "ఆరోగ్యకరమైన ఆకు",
            "symptoms": ["పచ్చని ఆకులు"],
            "solution": ["ఏ చర్య అవసరం లేదు"],
            "advice": ["మంచి సంరక్షణ"]
        },
        "Marathi": {
            "name": "निरोगी पान",
            "symptoms": ["हिरवी पाने"],
            "solution": ["काही गरज नाही"],
            "advice": ["काळजी घ्या"]
        }
    }
}

# -------------------------------
# UPLOAD IMAGE
# -------------------------------
st.subheader("📥 Upload Image")
file = st.file_uploader("Choose Image", type=["jpg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image")

    if st.button("🚀 Detect"):
        img_np = np.array(img)
        img_np = img_np[:, :, ::-1]

        results = model(img_np)
        res_img = results[0].plot()
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

        st.image(res_img, caption="Result")

        boxes = results[0].boxes

        if boxes and len(boxes) > 0:
            best = max(boxes, key=lambda x: float(x.conf[0]))
            label = model.names[int(best.cls[0])]
            conf = float(best.conf[0])

            st.success(f"Detected: {label} ({conf:.2f})")

            # -------------------------------
            # LANGUAGE SELECT
            # -------------------------------
            lang = st.selectbox("Select Language", ["English", "Telugu", "Marathi"])

            key = label.lower().replace("_deficiency", "")

            if key in data:
                info = data[key][lang]

                st.markdown(f"## 🌿 {info['name']}")

                st.markdown("### 🔍 Symptoms")
                for s in info["symptoms"]:
                    st.write("•", s)

                st.markdown("### 💊 Solution")
                for s in info["solution"]:
                    st.write("•", s)

                st.markdown("### 📌 Advice")
                for a in info["advice"]:
                    st.write("•", a)
            else:
                st.warning("No detailed info available")

        else:
            st.warning("No detection ❌")

else:
    st.info("Upload image first")
