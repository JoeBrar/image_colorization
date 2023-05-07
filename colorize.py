import streamlit as st
import numpy as np
import argparse
import cv2
import os
import requests
from skimage import color

file_url = "https://drive.google.com/uc?export=download&id=1bxg9ZobZp3ZZBixthtEOhSlofIo5Ghz7"
MODEL = requests.get(file_url)
print(MODEL)
DIR = r"D:\B.Tech Studies\auto colorization project\Auto colorization source code"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
#MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Argparser
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", type=str, required=True,
	#help="path to input black and white image")
#args = vars(ap.parse_args())
#file1=None

st.write("Convert B&W to Colored :")
file1 = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file1Uploader")
if file1 is not None:
	img1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
	img2 = cv2.imdecode(img1, 1)
	image=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
	net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
	pts = np.load(POINTS)

	class8 = net.getLayerId("class8_ab")
	conv8 = net.getLayerId("conv8_313_rh")
	pts = pts.transpose().reshape(2, 313, 1, 1)
	net.getLayer(class8).blobs = [pts.astype("float32")]
	net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

	#imgPath = r'D:\B.Tech Studies\auto colorization project\Colorizing-black-and-white-images-using-Python-master\images\einstein.jpg'
	#image = cv2.imread(imgPath)
	scaled = image.astype("float32") / 255.0
	lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

	resized = cv2.resize(lab, (224, 224))
	L = cv2.split(resized)[0]
	L -= 50
		
	print("Colorizing the image")
	net.setInput(cv2.dnn.blobFromImage(L))
	ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

	ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

	L = cv2.split(lab)[0]
	colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

	colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
	colorized = np.clip(colorized, 0, 1)

	colorized = (255 * colorized).astype("uint8")


	colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
	st.image(image, caption='Original Image', use_column_width=True)
	st.image(colorized_rgb, caption='Colorized Image', use_column_width=True)

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("Convert Colored to B&W :")

file2 = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file2Uploader")
if file2 is not None:
	print("ok file2")
	img_1 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
	img_2 = cv2.imdecode(img_1, 1)
	original_img=cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
	greyscale_img=cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
	st.image(original_img, caption='Original Image', use_column_width=True)
	st.image(greyscale_img, caption='Black and White Image', use_column_width=True)



