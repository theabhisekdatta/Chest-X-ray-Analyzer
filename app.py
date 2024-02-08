from fastai.vision.all import *
import streamlit as st

st.title(":orange[Chest X-ray Analyzer]")

uploaded_files = st.file_uploader(
    "Upload a Chest Xray Scan Photos", accept_multiple_files=True)


learn = load_learner('image_classification_model_vgg16.pkl')

for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    pedict = learn.predict(bytes_data)
    st.write(pedict)


# pedict = learn.predict("val/NORMAL/NORMAL2-IM-1442-0001.jpeg")

# print(pedict)
