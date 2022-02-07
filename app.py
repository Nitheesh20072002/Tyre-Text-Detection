import streamlit as st
from PIL import Image
import cv2
import numpy as np
from numpy import asarray
import easyocr as ocr

 

 
@st.cache
def load_image(img_file):
	img = Image.open(img_file)
	return img

 
@st.cache
def load_model(): 
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader

st.title("Tyre Text Extraction")

img_file = st.file_uploader("Upload tyre Images", type=['png','jpeg','jpg'])

 
reader = load_model()



if(img_file is not None):
	img = load_image(img_file)
	img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

	max_val=255
	th = 125
	block_size = 413
	constant = 2
	kernal = np.ones((3,3), np.uint8)
	erosion = cv2.erode(np.array(img), kernal, iterations=1)
	guassian = cv2.GaussianBlur(erosion, (5,5),cv2.BORDER_DEFAULT)
	ret, o5 = cv2.threshold(np.array(img), th, max_val, cv2.THRESH_TRUNC)
	_,t = cv2.threshold(guassian, 90, max_val, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
	mean = cv2.adaptiveThreshold(guassian, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)


	
	with st.spinner("🤖 AI is at Work! "):
		result = reader.readtext(np.array(o5))
		result_text = ''  
		p = []
		for text in result:
 
			if(text[-1]>0.3):
				result_text += ' '+text[1]
				p.append(text[-1])
 
		p = int(max(p)*100)

	st.image(img,width=250,caption="Original image")
	st.image(t,width=250,caption="preprocessed image")	 
	st.success("Predicted text: "+ result_text)
	st.success("Accuracy: " + str(p) + "%")
	
	 
 
	

	
	  


 



 

 
