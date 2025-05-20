import streamlit as st
import numpy as np
from PIL import Image
import os
import tempfile
from recommender import recommend

st.set_page_config(page_title="Sneaker Visual Recommender", layout="centered")
st.title("👟 Sneaker Visual Recommender")
st.write("Upload an image of a sneaker and get visually similar results.")

# Upload image
uploaded_file = st.file_uploader("Upload a sneaker image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_image_path = tmp.name

    # Show uploaded image
    st.image(temp_image_path, caption="Query Image", use_container_width=True)

    # Get recommendations
    results = recommend(temp_image_path, top_k=5)

    st.subheader("🔎 Top 5 Similar Sneakers:")
    cols = st.columns(5)  # Create 5 columns

    for i, (path, score) in enumerate(results):
        with cols[i]:
            try:
                img = Image.open(path)
                st.image(img, use_container_width=True)
                st.markdown(
                    f"<div style='text-align: center; font-size: 13px;'>{os.path.basename(path)}<br>Similarity: {score:.2f}</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Couldn't load image: {e}")