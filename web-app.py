import streamlit as st
import base64
from Main import predict_emotion_from_uploaded_file

st.set_page_config(page_title="Speech Emotion Recognizer", layout="wide")

# ✅ Remove only the Streamlit blue navbar
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ✅ Set background image
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(10,10,20,0.75), rgba(10,10,20,0.75)),
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            font-family: 'Segoe UI', sans-serif;
        }}
        h1 {{
            color: #A8DADC;
            text-align: center;
        }}
        .block {{
            background-color: rgba(40, 40, 60, 0.4);
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 0 15px rgba(0,0,0,0.3);
            margin-top: 10px;
        }}
        .usecase-text {{
            color: #FFDDD2;
            font-size: 15px;
        }}
        .app-text {{
            color: #E0F7FA;
            font-size: 15px;
        }}
        .tech-text {{
            color: #D0F4DE;
            font-size: 15px;
        }}
        .stButton > button {{
            background-color: #00B4D8;
            color: black;
            font-weight: bold;
            border-radius: 8px;
        }}
        .prediction-success {{
            background-color: rgba(72, 202, 228, 0.15);
            padding: 12px;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 18px;
            font-weight: bold;
            color: #ADE8F4;
        }}
        </style>
    """, unsafe_allow_html=True)

# ✅ Apply background image
set_background("Bg_image.jpg")

# ✅ Create 3-column layout
col1, col2, col3 = st.columns([1.5, 2.5, 1.5])

# 🟣 Left Column – Use Cases
with col1:
    st.markdown("<div class='block usecase-text'>", unsafe_allow_html=True)
    st.markdown("###   Use Cases of Emotion Classification")
    st.markdown("""
    - 🤖 **Human-Computer Interaction (HCI)** – Responsive interfaces.
    - 🧠 **Mental Health Monitoring** – Emotional distress detection.
    - 📞 **Customer Support** – Identifying frustration/satisfaction.
    - 🎓 **E-learning Platforms** – Adapting to student emotion.
    - 🚗 **Driver Monitoring** – Fatigue/stress detection on road.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 🔵 Center Column – Main App
with col2:
    st.markdown("<div class='block app-text'>", unsafe_allow_html=True)
    st.markdown("<h1>🎧 Speech Emotion Recognizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload a <code>.wav</code> file and discover the emotion behind the voice.</p>", unsafe_allow_html=True)

    st.markdown("#### 🎵 Upload a WAV file")
    uploaded_file = st.file_uploader("Choose a file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        emotion = predict_emotion_from_uploaded_file(uploaded_file)
        st.markdown(f"<div class='prediction-success'>🎯 Predicted Emotion: <strong>{emotion.capitalize()}</strong></div>", unsafe_allow_html=True)
    else:
        st.info("👋 Please upload a .wav file to continue.")
    st.markdown("</div>", unsafe_allow_html=True)

# 🟠 Right Column – Tech Stack
with col3:
    st.markdown("<div class='block tech-text'>", unsafe_allow_html=True)
    st.markdown("###   Tech Stack Used")
    st.markdown("""
    - 🔥 **PyTorch** – For deep learning model.
    - 🎼 **Librosa** – Audio feature extraction.
    - 🧱 **Streamlit** – Interactive web UI.
    - 🧠 **CNN + BiLSTM + Attention** – Emotion classifier.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center style='color: #AAAAAA;'>Made with ❤️ using PyTorch, Librosa & Streamlit</center>", unsafe_allow_html=True)



#Emotion ( 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).