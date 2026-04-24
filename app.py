import numpy as np
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Leaf Detection 🌿",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# UI DESIGN (UNCHANGED)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background:#000;
    # background: linear-gradient(120deg, #141e30, #243b55);
    color: #ffffff;
}
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: #fff;
}
.sub-title {
    text-align: center;
    color: #cccccc;
}
.main-box {
    background: rgba(255,255,255,0.07);
    padding: 25px;
    border-radius: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #00e6b8, #00b3ff);
    color: black;
    border-radius: 10px;
    width: 100%;
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
# DATA (UNCHANGED)
# ------------------------------
deficiency_info = {

    "nitrogen": {
        "English": {
            "name": "Nitrogen Deficiency",
            "description": [
                "Nitrogen is essential for chlorophyll production",
                "Supports overall plant growth and leaf development",
                "Helps in photosynthesis and energy formation",
                "Deficiency mainly affects older leaves first",
                "Leads to weak and stunted plant growth"
            ],
            "symptoms": [
                "Yellowing of older leaves",
                "Slow and stunted growth",
                "Thin and weak stems"
            ],
            "solution": [
                "Apply urea or ammonium fertilizers",
                "Use organic compost or manure",
                "Apply nitrogen-rich biofertilizers"
            ],
            "advice": [
                "Maintain soil fertility",
                "Perform regular soil testing",
                "Avoid excessive watering"
            ]
        },

        "Marathi": {
            "name": "नायट्रोजन कमतरता",
            "description": [
                "नायट्रोजन हरितद्रव्य निर्मितीसाठी आवश्यक आहे",
                "झाडाच्या वाढीस मदत करते",
                "प्रकाशसंश्लेषणासाठी महत्त्वाचे आहे",
                "जुनी पाने आधी प्रभावित होतात",
                "झाडाची वाढ मंदावते"
            ],
            "symptoms": ["जुनी पाने पिवळी होणे", "हळू वाढ", "कमकुवत खोड"],
            "solution": ["युरिया वापरा", "सेंद्रिय खत वापरा"],
            "advice": ["माती तपासणी करा", "जास्त पाणी टाळा"]
        },

        "Telugu": {
            "name": "నైట్రోజన్ లోపం",
            "description": [
                "నైట్రోజన్ క్లోరోఫిల్ తయారీకి అవసరం",
                "మొక్క ఎదుగుదలకు సహాయపడుతుంది",
                "ఫోటోసింథసిస్‌కు ముఖ్యమైనది",
                "ముందుగా పాత ఆకులు ప్రభావితం అవుతాయి",
                "మొక్క ఎదుగుదల తగ్గుతుంది"
            ],
            "symptoms": ["పాత ఆకులు పసుపు", "నెమ్మదిగా ఎదుగుదల", "బలహీనమైన కాండం"],
            "solution": ["యూరియా వేయండి", "సేంద్రీయ ఎరువులు వాడండి"],
            "advice": ["మట్టి పరీక్ష చేయండి", "అధిక నీరు నివారించండి"]
        }
    },

    "kalium": {
        "English": {
            "name": "Potassium (Kalium) Deficiency",
            "description": [
                "Potassium regulates water balance in plants",
                "Improves disease resistance",
                "Supports enzyme activity",
                "Helps in strong stem development",
                "Deficiency weakens plant immunity"
            ],
            "symptoms": [
                "Brown or burnt leaf edges",
                "Weak stems",
                "Poor resistance to disease"
            ],
            "solution": [
                "Apply potash fertilizers",
                "Use balanced NPK fertilizers"
            ],
            "advice": [
                "Avoid excess nitrogen usage",
                "Maintain proper irrigation"
            ]
        },

        "Marathi": {
            "name": "पोटॅशियम कमतरता",
            "description": [
                "पोटॅशियम पाण्याचे संतुलन राखते",
                "रोगप्रतिकारक शक्ती वाढवते",
                "खोड मजबूत करते",
                "एन्झाइम क्रिया सुधारते",
                "कमतरतेमुळे झाड कमजोर होते"
            ],
            "symptoms": ["पानांच्या कडा तपकिरी", "कमकुवत खोड"],
            "solution": ["पोटॅश खत वापरा"],
            "advice": ["संतुलित खत वापरा"]
        },

        "Telugu": {
            "name": "పొటాషియం లోపం",
            "description": [
                "పొటాషియం నీటి నియంత్రణకు సహాయపడుతుంది",
                "రోగ నిరోధక శక్తి పెంచుతుంది",
                "కాండాన్ని బలంగా చేస్తుంది",
                "ఎంజైమ్ క్రియలను మెరుగుపరుస్తుంది",
                "లోపం ఉన్నప్పుడు మొక్క బలహీనమవుతుంది"
            ],
            "symptoms": ["ఆకుల అంచులు గోధుమ", "బలహీనమైన కాండం"],
            "solution": ["పొటాష్ ఎరువు వేయండి"],
            "advice": ["సమతుల్య ఎరువులు వాడండి"]
        }
    },

    "boron": {
        "English": {
            "name": "Boron Deficiency",
            "description": [
                "Boron is essential for cell division",
                "Helps in flower and fruit development",
                "Supports growth of new leaves",
                "Affects growing points of the plant",
                "Deficiency leads to improper development"
            ],
            "symptoms": [
                "Distorted or curled leaves",
                "Thick and brittle leaves",
                "Poor flowering",
                "Fruit deformation"
            ],
            "solution": [
                "Apply boron spray in low concentration",
                "Use micronutrient fertilizers",
                "Apply recommended dosage only"
            ],
            "advice": [
                "Avoid over-irrigation",
                "Do not overapply boron",
                "Monitor plant growth regularly"
            ]
        },

        "Marathi": {
            "name": "बोरॉन कमतरता",
            "description": [
                "बोरॉन पेशी विभाजनासाठी आवश्यक आहे",
                "फुल आणि फळ विकासासाठी मदत करते",
                "नवीन पानांच्या वाढीस मदत करते",
                "वाढीच्या भागांवर परिणाम होतो",
                "कमतरतेमुळे वाढ बिघडते"
            ],
            "symptoms": ["वाकडी पाने", "फळांची कमी वाढ"],
            "solution": ["बोरॉन स्प्रे वापरा"],
            "advice": ["जास्त पाणी टाळा"]
        },

        "Telugu": {
            "name": "బోరాన్ లోపం",
            "description": [
                "బోరాన్ కణాల అభివృద్ధికి అవసరం",
                "పుష్పాలు మరియు పండ్ల అభివృద్ధికి సహాయపడుతుంది",
                "కొత్త ఆకుల పెరుగుదలకు సహాయపడుతుంది",
                "మొక్క ఎదుగుదల భాగాలను ప్రభావితం చేస్తుంది",
                "లోపం ఉన్నప్పుడు అభివృద్ధి తగ్గుతుంది"
            ],
            "symptoms": ["వంకర ఆకులు", "పండ్ల ఎదుగుదల తగ్గుతుంది"],
            "solution": ["బోరాన్ స్ప్రే చేయండి"],
            "advice": ["అధిక నీరు ఇవ్వవద్దు"]
        }
    },

    "mg": {
        "English": {
            "name": "Magnesium Deficiency",
            "description": [
                "Magnesium is part of chlorophyll",
                "Essential for photosynthesis",
                "Helps in energy production",
                "Supports enzyme activation",
                "Deficiency reduces plant growth"
            ],
            "symptoms": [
                "Yellowing between veins",
                "Leaf curling",
                "Reduced growth"
            ],
            "solution": [
                "Apply magnesium sulfate",
                "Use dolomite lime if needed"
            ],
            "advice": [
                "Maintain soil pH",
                "Ensure balanced nutrients"
            ]
        },

        "Marathi": {
            "name": "मॅग्नेशियम कमतरता",
            "description": [
                "मॅग्नेशियम हरितद्रव्याचा भाग आहे",
                "प्रकाशसंश्लेषणासाठी आवश्यक आहे",
                "ऊर्जा निर्मितीस मदत करते",
                "एन्झाइम क्रिया सुधारते",
                "कमतरतेमुळे वाढ कमी होते"
            ],
            "symptoms": ["शिरांमध्ये पिवळेपणा", "पाने वाकणे"],
            "solution": ["मॅग्नेशियम सल्फेट वापरा"],
            "advice": ["मातीचा pH सांभाळा"]
        },

        "Telugu": {
            "name": "మెగ్నీషియం లోపం",
            "description": [
                "మెగ్నీషియం క్లోరోఫిల్‌లో భాగం",
                "ఫోటోసింథసిస్‌కు అవసరం",
                "శక్తి ఉత్పత్తికి సహాయపడుతుంది",
                "ఎంజైమ్ క్రియలను మెరుగుపరుస్తుంది",
                "లోపం ఉన్నప్పుడు ఎదుగుదల తగ్గుతుంది"
            ],
            "symptoms": ["శిరల మధ్య పసుపు", "ఆకులు ముడుచుకోవడం"],
            "solution": ["మెగ్నీషియం సల్ఫేట్ వేయండి"],
            "advice": ["మట్టి pH సరిచూడండి"]
        }
    },

    "healthy": {
        "English": {
            "name": "Healthy Leaf",
            "description": [
                "Leaves are bright green in color",
                "Plant shows normal growth",
                "No nutrient deficiency symptoms",
                "Strong stems and structure",
                "Proper photosynthesis occurs"
            ],
            "symptoms": [
                "Green leaves",
                "Normal growth",
                "No spots or damage"
            ],
            "solution": ["No action required"],
            "advice": [
                "Continue proper care",
                "Maintain balanced fertilizers",
                "Monitor regularly"
            ]
        },

        "Marathi": {
            "name": "निरोगी पान",
            "description": [
                "पाने हिरवी असतात",
                "झाडाची वाढ सामान्य असते",
                "कोणतीही कमतरता नसते"
            ],
            "symptoms": ["हिरवी पाने", "सामान्य वाढ"],
            "solution": ["काही गरज नाही"],
            "advice": ["चांगली काळजी घ्या"]
        },

        "Telugu": {
            "name": "ఆరోగ్యకరమైన ఆకు",
            "description": [
                "ఆకులు పచ్చగా ఉంటాయి",
                "మొక్క ఎదుగుదల సాధారణంగా ఉంటుంది",
                "ఏ లోపం ఉండదు"
            ],
            "symptoms": ["పచ్చని ఆకులు", "సాధారణ ఎదుగుదల"],
            "solution": ["ఏ చర్య అవసరం లేదు"],
            "advice": ["మంచి సంరక్షణ కొనసాగించండి"]
        }
    }
}
# -------------------------------
# SESSION STATE
# -------------------------------
if "detected_class" not in st.session_state:
    st.session_state["detected_class"] = None

if "detected_conf" not in st.session_state:
    st.session_state["detected_conf"] = None
# -------------------------------
# STEP 1: INPUT
# -------------------------------
st.markdown("## 📥Upload or 📷Capture Image")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:
        st.session_state["input_image"] = Image.open(uploaded_file)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)

    cam_on = st.toggle("Enable Camera")

    if cam_on:
        camera_image = st.camera_input("Capture Image")

        if camera_image:
            st.session_state["input_image"] = Image.open(camera_image)
            st.success("Image Captured ✅")

    else:
        st.info("Camera OFF")

    st.markdown("</div>", unsafe_allow_html=True)
# -------------------------------
# STEP 2: DETECTION
# -------------------------------
st.markdown("## 🧠Detection Result")

input_img = st.session_state.get("input_image", None)

if input_img:

    if st.button("🚀 Detect Now"):

        img_np = np.array(input_img)
        img_np = img_np[:, :, ::-1]
        img_np = cv2.resize(img_np, (640, 640))
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

        results = model(img_np, conf=0.6)

        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        colA, colB = st.columns(2)

        with colA:
            st.image(input_img, caption="Original")

        with colB:
            st.image(result_img, caption="Detected")

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            st.warning("No detection 🚫")

        else:
            best_box = max(boxes, key=lambda x: float(x.conf[0]))

            st.session_state["detected_conf"] = float(best_box.conf[0])
            st.session_state["detected_class"] = model.names[int(best_box.cls[0])]

            st.success(f"Detected: {st.session_state['detected_class']}")

else:
    st.info("Upload image first")
# -------------------------------
# STEP 3: LANGUAGE (AFTER DETECTION)
# -------------------------------
if st.session_state["detected_class"]:

    st.markdown("## 🌐 Select Language")

    language = st.selectbox(
        "Choose Language",
        ["en", "mr", "te"],
        format_func=lambda x: {
            "en": "English",
            "mr": "Marathi",
            "te": "Telugu"
        }[x]
    )

    detected_class = st.session_state["detected_class"]

    # ✅ FIX: normalize class name
    detected_class = detected_class.lower().replace("_deficiency", "").strip()

    info = deficiency_info.get(detected_class)

    if info:
        # ✅ FIX: language mapping
        lang_map = {
            "en": "English",
            "mr": "Marathi",
            "te": "Telugu"
        }

        selected_lang = lang_map.get(language, "English")

        # ✅ FIX: correct data access
        data = info.get(selected_lang, info["English"])

        st.markdown(f"## 🌿 {data['name']}")

        # ✅ FIXED HERE (no index numbers now)
        for d in data.get("description", []):
            st.write(f"• {d}")

        st.markdown("### 🌿 Symptoms")
        for s in data.get("symptoms", []):
            st.write(f"• {s}")

        st.markdown("### 💊 Solution")
        for s in data.get("solution", []):
            st.write(f"• {s}")

        st.markdown("### 📌 Advice")
        for a in data.get("advice", []):
            st.write(f"• {a}")

    else:
        st.warning("Low confidence detection ⚠️")

else:
    st.info("👆 Upload or capture an image first")
