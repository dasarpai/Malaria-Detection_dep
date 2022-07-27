import tensorflow as tf
import streamlit as st 
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    loaded_model = tf.keras.models.load_model(r'model-Malaria-Detection-xception')
    # loaded_model.summary()
    #x = tf.random.uniform((10, 3))
    return loaded_model

file1 = st.file_uploader('Upload an Image')
btn = st.button('Submit Image for Prediction')
if btn and file1!=None:
    img = image.load_img(file1.name, target_size=(71, 71))
    print (img)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    loaded_model = load_model()
    prediction = loaded_model.predict(img_preprocessed)

    if prediction<.1:
        title = "Infected"
    else:
        title = "Uninfected"

    st.header("It is "+title)
    st.image(file1.name)
