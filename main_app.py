import numpy as np
import streamlit as st
import cv2
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import LSTM, Dense


model= load_model('dog_breed.h5')

class_names=['scottish_deerhound','maltese_dog'	,'afghan_hound','entlebucher']

st.title("Dog breed Prediction")
st.markdown("upload the image of  dog")

dog_image=st.file_uploader("choose a image ----", type="png")
submit= st.button("predict")


if submit:
    if dog_image is not None:

        #convert the file to the open cv image
        file_bytes=np.asanyarray(bytearray(dog_image.read(),dtype=np.unit8))
        opencv_image=cv2.imdecode(file_bytes,1)

#displaying the image
st.image(opencv_image,channels="RGB")
opencv_image=cv2.resize(opencv_image,(224,224))
opencv_image.shape=(1,224,224,3)
y_pred=model.predict(opencv_image)


st.title(str("the dog breed is " +class_names[np.argmax(y_pred)]))