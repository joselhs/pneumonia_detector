import keras
import numpy as np
import streamlit as st
from keras import layers, models, optimizers  # modeling
from PIL import Image

MODEL = "cnn_augmented_dropout_90_7.h5"


@st.cache(allow_output_mutation=True)
def load_model():
    print("loading model")
    model = keras.models.load_model(f"model/{MODEL}", compile=True)

    return model


def preprocess_image(img):
    print("preprocessing image")
    image = Image.open(img)
    print(image)
    p_img = image.resize((224, 224))

    return np.array(p_img)


def predict(model, img):
    print("Predicting result")
    print(f'shape: {img.shape}')

    prediction = model.predict(np.reshape(img, [1, 224, 224, 3]))

    print(prediction)
    return prediction

