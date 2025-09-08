import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import pandas as pd
import os
import base64

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Object Detection",
    page_icon="🔍",
    layout="centered"
)

# ----------------- Background Image -----------------
def set_background(image_file):
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
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, 
        .stApp h5, .stApp h6, .stApp p, 
        .stApp span, .stApp div, .stApp label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background_image.jpg")

# ----------------- Load YOLOv8m Model -----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")   # Medium model pretrained on COCO dataset (80 classes)

model = load_model()

# ----------------- Title & Sidebar -----------------
st.title("Object Detection System")
st.markdown(
    "Upload an **image** or try a demo sample. "
    "The AI will automatically detect objects (trained on the **COCO dataset: 80 classes**)."
)

st.sidebar.title("ℹ️ About this App")
st.sidebar.markdown("""
This demo uses **YOLOv8m** (medium model) - pretrained on the **COCO dataset (80 classes)**.
  
That means it can detect **people, vehicles, animals, everyday objects**, and more.""")  

st.sidebar.markdown("---")

st.sidebar.markdown("""
👨‍💻 **How it works:**  
- **Step 1:** Upload an image or use the demo image.  
- **Step 2:** The AI will highlight detected objects with bounding boxes.  
- **Step 3:** A chart shows how many of each object were detected.  

✅ Works with a wide range of everyday images.
""")

st.sidebar.info("📌 Tip: Try uploading images with people, animals, or street scenes to see diverse detections.")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Created by **Sankaran S**")


# ----------------- Main App -----------------
st.markdown("### 📂 Upload your own image **or** use demo sample")

choice = st.radio("Choose an option:", ["Upload File", "Use Demo Image"])
filepath, file_extension = None, None

if choice == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        filepath = tfile.name
        file_extension = uploaded_file.name.split(".")[-1].lower()
else:
    filepath = "test_image1.jpg"   
    file_extension = "jpg"

# ----------------- Detection -----------------
if filepath is not None and file_extension in ["jpg", "png", "jpeg"]:
    img = cv2.imread(filepath)

    # Run YOLO
    results = model(img)
    annotated_img = results[0].plot()

    # Extract detections
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        detections.append(label)

    if len(detections) > 0:
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

        st.markdown("✅ The chart shows the number of each type of object detected in the image.")
    else:
        st.warning("⚠️ No objects detected in this image. Try another one!")

    # Cleanup only if user uploaded
    if choice == "Upload File" and uploaded_file is not None:
        try:
            os.unlink(filepath)
        except PermissionError:
            pass
