import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from functions import losses
import config
from PIL import Image


@st.cache_resource
def load_model():
    return keras.models.load_model('seg_model.h5', custom_objects={"FocalLoss": losses.FocalLoss, "dice_coef": losses.dice_coef})


def ShowImage(model, src_img, one_image=False):

    image = Image.open(src_img)
    img = np.array(image, dtype=np.float64)
    img = img[::config.IMG_SCALING[0], ::config.IMG_SCALING[1]]
    img = img/255
    img = tf.expand_dims(img, axis=0)
    predicted = model.predict(img)
    predicted = np.squeeze(predicted, axis=0)

    if one_image:
        st.image(predicted, width=500)
    else:
        # rgb_img = cv2.imread(src_img)
        # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        image_list = []
        image_list.append(image)
        image_list.append(predicted)

        captions = ['Base image', 'Detected ships']
        st.image(image_list, caption=captions, clamp=True, width=225)


model = load_model()

st.title('Ship detection')
st.caption('This is model that can detecting ships on images.')

st.header('Select an image')

uploaded_image = st.file_uploader(
    "Select a photo", type=['png', 'jpg', 'jpeg'])
if st.button('Detect the ships'):
    if uploaded_image is not None:
        ShowImage(model, uploaded_image)

if st.button('Detect the ships (show only predicted image)'):
    if uploaded_image is not None:
        ShowImage(model, uploaded_image, True)
