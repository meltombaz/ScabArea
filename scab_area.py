import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from tempfile import TemporaryDirectory
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import math

st.set_page_config(layout="wide")
st.title("🧮 Dark Area Calculator with mm² Conversion")
st.write("Upload your `.tif` images and select a scale bar to convert pixel² to mm².")

# ----------------------------
# AREA CALCULATION FUNCTION
# ----------------------------
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

# ----------------------------
# SCALE BAR MEASUREMENT TOOL
# ----------------------------
st.subheader("📏 Step 1: Upload Reference Image for Scale Bar")
scale_img = st.file_uploader("Upload an image with a visible scale bar", type=["tif", "tiff", "png", "jpg"])

pixels_per_mm = None
mm2_conversion = 1.0

if scale_img:
    image = Image.open(scale_img)
    st.write("Click **exactly two points**: the ends of the scale bar.")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="#000000",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="transform",  # Disable drawing
        point_display_radius=5,
        key="canvas_scale_bar",
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) >= 2:
        # Extract the first two clicks
        p1 = canvas_result.json_data["objects"][0]["left"], canvas_result.json_data["objects"][0]["top"]
        p2 = canvas_result.json_data["objects"][1]["left"], canvas_result.json_data["objects"][1]["top"]
        pixel_distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        st.write(f"📐 Measured scale bar length: **{pixel_distance:.2f} pixels**")

        real_length = st.number_input("Enter real-world length of scale bar (in mm)", min_value=0.0001, value=1.0)
        pixels_per_mm = pixel_distance / real_length
        mm2_conversion = (1 / pixels_per_mm) ** 2
        st.success(f"1 pixel = {1 / pixels_per_mm:.5f} mm → Conversion active ✅")
    else:
        st.info("Please click two points on the image to define the scale bar.")

# ----------------------------
# IMAGE ANALYSIS SECTION
# ----------------------------
st.subheader("🖼️ Step 2: Upload Images to Analyze")
uploaded_files = st.file_uploader(
    "Upload one or more `.tif` images to analyze dark areas", type=["tif", "tiff"], accept_multiple_files=True
)

threshold_value = st.slider("Threshold for detecting dark objects", 0, 255, 50)

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
                    "Area (px²)": round(pixel_area, 2),
                    "Area (mm²)": round(area_mm2, 4)
                })
            except Exception as e:
                results.append({"Image": file.name, "Area (px²)": "Error", "Area (mm²)": str(e)})

    df = pd.DataFrame(results)
    st.subheader("📊 Measurement Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", csv, "dark_area_measurements.csv", "text/csv")
