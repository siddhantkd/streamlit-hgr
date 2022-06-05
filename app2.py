# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:45:42 2022

@author: siddh
"""

from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from classification_models.keras import Classifiers

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate, Conv2DTranspose, Add, AveragePooling2D
from tensorflow.keras import models
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras.layers import *

import segmentation_models as sm

input_size = (320, 320, 3)
num_class = 10

model = keras.models.load_model("fyhgr.h5")
#model.summary()

# Specify Image Dimensions
IMG_WIDTH = 320
IMG_HEIGHT = 320
IMG_CHANNELS = 3


class VideoProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")
		img = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
		img = np.expand_dims(img, axis=0)

		y_pred = model.predict(img, verbose=0)
		y_classes = [np.argmax(y_pred[0])]
		classes_cats = ["A", "B", "C", "D", "E", "F", "H", "I", "J", "K"]

		cv2.putText(frm, "CLASSIFICATION RESULT :", (3, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.putText(frm, str(classes_cats[y_classes[-1]]), (500, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

		return av.VideoFrame.from_ndarray(frm, format='bgr24')


webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
				)
				)