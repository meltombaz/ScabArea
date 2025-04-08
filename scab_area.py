import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("笨蛋 Scab Area Calculator (with mm² Conversion)")
st.write("Upload .tif images and optionally a reference image with a scale bar to convert from pixel² to mm².")

# --- Load and calculate area ---
def calculate_dark_area(image_path, threshold_value=50):
    image = io.imread(image_path)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)

    return total_area

# --- Scale Reference Upload ---
st.sidebar.header("📏 Scale Bar Calibration")
scale_img = st.sidebar.file_uploader("Upload a reference image with scale bar", type=["tif", "tiff", "png", "jpg"])

scale_bar_length = None
pixels_per_mm = None
mm2_conversion = 1

if scale_img:
    st.sidebar.image(scale_img, caption="Scale bar image", use_column_width=True)
    st.sidebar.info("Click two points on the image where the scale bar starts and ends.")
    
    # Let user click two points
    image_pil = Image.open(scale_img)
    fig = px.imshow(image_pil)
    fig.update_layout(dragmode='drawopenpath')
    draw_fig = st.sidebar.plotly_chart(fig, use_container_width=True)

    with st.sidebar.form("scale_form"):
        real_world_length = st.number_input("Real length of selected bar (mm)", min_value=0.0001, value=1.0)
        pixel_length = st.number_input("Pixel length of bar (manually measured)", min_value=1.0, value=100.0)
        submitted = st.form_submit_button("Calibrate")
    
    if submitted:
        pixels_per_mm = pixel_length / real_world_length
        mm2_conversion = (1 / pixels_per_mm)**2
        st.sidebar.success(f"Scale set: 1 pixel = {1/pixels_per_mm:.4f} mm")

# --- Main Area: Upload .tif images ---
st.subheader("📂 Upload your .tif images")
uploaded_files = st.file_uploader(
    "Upload one or more .tif images", type=["tif", "tiff"], accept_multiple_files=True
)

threshold_value = st.slider("Threshold for dark detection", 0, 255, 50)

# --- Process each image ---
if uploaded_files:
    results = []

    with TemporaryDirectory() as tempdir:
        for file in uploaded_files:
            file_path = Path(tempdir) / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())
            try:
                pixel_area = calculate_dark_area(str(file_path), threshold_value)
                area_mm2 = pixel_area * mm2_conversion
                results.append({
                    "Image": file.name,
                    "Area_px²": round(pixel_area, 2),
                    "Area_mm²": round(area_mm2, 4)
                })
            except Exception as e:
                results.append({"Image": file.name, "Area_px²": "Error", "Area_mm²": str(e)})

    df = pd.DataFrame(results)
    st.subheader("📊 Measurement Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv, "dark_area_measurements.csv", "text/csv")
