import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt # plots

from PIL import Image # image
from keras.preprocessing.image import ImageDataGenerator # pre-processing
from keras import layers, models, optimizers # modeling

# App Title
st.title("Pneumodetector APP")

# Introduction text
st.markdown(unsafe_allow_html=True, body="<p>Welcome to Pneumodetector APP. </p>"
            "<p>This is a basic app built with Streamlit."
            "With this app, you can upload a Chest X-Ray image and predict if the patient "
            "from that image suffers pneumonia or not.</p>"
            "<p>The model used is a Convolutional Neural Network (CNN) and in this moment has a test accuracy of "
                                         "<strong>90.7%.</strong></p>")


st.markdown("First, let's load an X-Ray Chest image.")

# Img uploader
img = st.file_uploader(label="Load X-Ray Chest image", type=['jpeg', 'jpg', 'png'], key="xray")


if img is not None:
    image = np.array(Image.open(img))

    if st.checkbox('Show image'):
        st.image(image, use_column_width=True)

    # TODO: show error if there was a problem uploading image


