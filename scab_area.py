import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from tempfile import TemporaryDirectory

st.title("Scab Area Calculator in Images 💗")
st.write("This app calculates the area of scabs in `.tif` images.")

def calculate_dark_area(image_path, threshold_value=50):
    image = io.imread(image_path)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)

    return total_area

uploaded_files = st.file_uploader(
    "Upload multiple .tif images", type=["tif", "tiff"], accept_multiple_files=True
)

threshold_value = st.slider("Threshold value for dark detection", min_value=0, max_value=255, value=50)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")
    results = []

    with TemporaryDirectory() as tempdir:
        for file in uploaded_files:
            file_path = os.path.join(tempdir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            
            try:
                area = calculate_dark_area(file_path, threshold_value=threshold_value)
                results.append({"Image": file.name, "DarkArea_px2": area})
            except Exception as e:
                results.append({"Image": file.name, "DarkArea_px2": f"Error: {e}"})

    df = pd.DataFrame(results)
    st.subheader("📊 Area Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", csv, "dark_area_measurements.csv", "text/csv")
