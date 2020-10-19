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

    image = Image.open(img).convert("RGB")
    print(image.getdata().mode)
    print(image)
    p_img = image.resize((224, 224))

    return np.array(p_img) / 255.0


def predict(model, img):
    print("Predicting result")
    print(f'shape: {img.shape}')

    prob = model.predict(np.reshape(img, [1, 224, 224, 3]))
    print(prob)
    if prob > 0.5:
        prediction = True
    else:
        prediction = False

    print(prediction)
    return prediction

