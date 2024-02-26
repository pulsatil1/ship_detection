import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from functions import losses
import config
from PIL import Image
from skimage.io import imread
from skimage.morphology import binary_opening


@st.cache_resource
def load_model():
    return keras.models.load_model('fullres_model.h5', custom_objects={"dice_loss": losses.dice_loss})


def ShowImage(model, src_img, one_image=False):

    image = Image.open(src_img)
    img = np.array(image, dtype=np.float64)
    img = np.resize(img, (768, 768, 3))
    img = img/255
    img = tf.expand_dims(img, axis=0)
    predicted = model.predict(img)
    predicted = np.squeeze(predicted, axis=0)

    if one_image:
        st.image(predicted, width=500)
    else:
        image_list = []
        image_list.append(image)
        image_list.append(predicted)

        captions = ['Base image', 'Prediction']
        st.image(image_list, caption=captions, clamp=True, width=225)


model = load_model()

st.title('Ship detection')
st.caption(
    'Here using the semantic segmentation model that can segment ships on images.')

st.header('Select an image')

uploaded_image = st.file_uploader(
    "Select an image", type=['png', 'jpg', 'jpeg'])
if st.button('Detect the ships'):
    if uploaded_image is not None:
        ShowImage(model, uploaded_image)

if st.button('Show only predicted image'):
    if uploaded_image is not None:
        ShowImage(model, uploaded_image, True)
