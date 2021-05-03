import os
import streamlit as st
import cv2
import numpy as np
import keras
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np 
from PIL import Image,ImageOps 

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

def app():
	new_model = keras.models.load_model("modelpncnn.h5")
	st.title("Pneumonia Detection")
	st.markdown(hide_streamlit_style, unsafe_allow_html=True);
	
	uploaded_file_hem = st.file_uploader("Choose a image file", type=['png','jpg','jpeg'])
	if uploaded_file_hem is not None:
		image = Image.open(uploaded_file_hem)
		size =(150,150)
		image1=image.resize(size)
		image1=ImageOps.grayscale(image1)
		x=np.asarray(image1)
		x=np.expand_dims(x,axis=0)
		x=np.reshape(x,(-1,150,150,1))
		x=x/255.0
		if(new_model.predict_classes(x)[0][0]==1):
			st.write('***Result: Normal***')
			with st.beta_expander('Show Image'):
				st.image(image,channels='BGR',width=200)
		else:
			st.write('***Result: Pneumonia***')
			with st.beta_expander('Show Image'):
				st.image(image,channels='BGR',width=200)
				
if __name__ == '__main__':
	app()
