import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO
import pandas as pd
import os

st.set_page_config(page_title="Object Detection", page_icon="🚗", layout="centered")

def set_background_image(image_file):
    import base64
    import streamlit as st
    
    with open(image_file, "rb") as f:
        img_data = f.read()
    img_base64 = base64.b64encode(img_data).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: 
                linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        /* Make all text bright white */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, 
        .stApp h5, .stApp h6, .stApp p, 
        .stApp span, .stApp div, .stApp label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background_image('background_image.jpg')

# ✅ Cache the model (loads only once)
@st.cache_resource
def load_model():
    return YOLO("yolov8l.pt")

model = load_model()

# App Title
st.title("🚗 Object Detection & Recognition System")
st.markdown("Upload an image or video, or try demo samples. The system will automatically detect objects & specially built for vehicles using **AI-powered YOLOv8**.")

# ---------------- Sidebar ----------------
st.sidebar.title("ℹ️ About this App")
st.sidebar.markdown("""
This system is built to **automatically detect vehicles** from uploaded images or videos using **AI (YOLOv8)**.

👨‍✈️ **How it works (simple terms):**
1. **Upload** an image or video of vehicles, or select a demo sample.
2. The AI model will **scan and highlight** vehicles on screen.
3. For **images**, you will also see a chart showing how many vehicles were detected.

✅ Designed to be beginner-friendly so that **non-technical officers** can easily use it.
""")

st.sidebar.markdown("---")
st.sidebar.info("📌 Tip: For best results, upload clear videos or images (daylight, good angle).")

# ---------------- Main App ----------------
st.markdown("### 📂 Upload your own file **or** try demo samples")

choice = st.radio("Choose an option:", ["Upload File", "Use Demo File"])

uploaded_file = None
filepath = None

if choice == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload Video or Image", type=["mp4", "avi", "mov", "jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Save upload to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        tfile.close()
        filepath = tfile.name
        file_extension = uploaded_file.name.split(".")[-1].lower()
else:
    demo_type = st.selectbox("Select a demo sample:", ["Demo Image", "Demo Video"])
    if demo_type == "Demo Image":
        filepath = "test_image1.jpg"   
        file_extension = "jpg"
    else:
        filepath = "test_video2.mp4"   
        file_extension = "mp4"

# ---------------- Processing ----------------
if filepath is not None:

    # If image uploaded or demo
    if file_extension in ["jpg", "png", "jpeg"]:
        img = cv2.imread(filepath)

        # Detect objects
        results = model(img)
        annotated_img = results[0].plot()

        # Extract detections
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append(label)

        df = pd.DataFrame(detections, columns=["Object"])
        counts = df["Object"].value_counts().reset_index()
        counts.columns = ["Object", "Count"]

        # Layout: image left, chart right
        col1, col2 = st.columns(2)

        with col1:
            st.image(annotated_img, channels="BGR", caption="🖼️ Processed Image with Detected Objects")

        with col2:
            st.markdown("### 📊 Objects Detected (Counts)")
            st.bar_chart(counts.set_index("Object"))

        st.markdown("✅ The chart shows the **number of each type of object** detected in the uploaded/demo image.")

    # If video uploaded or demo
    else:
        cap = cv2.VideoCapture(filepath)

        fps = 24
        frame_delay = 1.0 / fps  

        stframe = st.empty()
        timestamp_placeholder = st.empty()
        status_placeholder = st.empty()

        status_placeholder.markdown("🎥 **Processing Video... Detecting Vehicles Frame by Frame**")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            # Show video frames
            stframe.image(annotated_frame, channels="BGR")

            # Show timestamp
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  
            timestamp_placeholder.write(f"⏱ Current Time: **{timestamp:.2f} sec**")

            time.sleep(frame_delay)

        cap.release()

        status_placeholder.markdown("✅ **Processing complete! Video finished.**")

    # Cleanup only for user-uploaded files
    if choice == "Upload File" and uploaded_file is not None:
        try:
            os.unlink(filepath)
        except PermissionError:
            pass
