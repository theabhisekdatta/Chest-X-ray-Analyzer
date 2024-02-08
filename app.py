from fastai.vision.all import *
import streamlit as st
from PIL import Image
from io import BytesIO  # Import BytesIO class from io module

# Title of the Web Application
st.title(":orange[Chest X-ray Analyzer]")

st.markdown("---")

# File uploader widget
uploaded_files = st.file_uploader(
    "Upload a Chest X-ray Scan Photo", accept_multiple_files=True)

# Load the trained model
learn = load_learner('image_classification_model_vgg16.pkl')

# Function to process uploaded files and make predictions


def process_files(uploaded_files):
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        img = Image.open(BytesIO(bytes_data))

        # Resize the image to a smaller size
        # img = img.resize((100, 100))

        st.image(img, caption='Uploaded Image', use_column_width=True)

        with st.spinner(f'Analyzing {uploaded_file.name}...'):
            prediction = learn.predict(bytes_data)
            st.write("The Result is : ", prediction[0])


# Adding a submit button to trigger predictions
if st.button('Submit'):
    if uploaded_files:
        process_files(uploaded_files)
    else:
        st.warning("Please upload an image before submitting.")
