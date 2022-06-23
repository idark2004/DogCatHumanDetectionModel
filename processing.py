from matplotlib.pyplot import axis

from numpy import imag
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn import preprocessing

input_shape = (224,224)

_model = load_model('../models/model.h5')

def read_image(image_encoded) :
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def preprocess(image : Image.Image):
    #resize
    image = image.resize(input_shape)
    image = np.array(image)
    image = image.astype('float64')
    #normalization
    image= (image - np.min(image)) / (np.max(image) - np.min(image))
    #convert to 4D
    image = np.expand_dims(image, axis=0)
    
    return image

def predict(image : np.ndarray):
    prediction = _model.predict(image)
    #Map label with predicted result
    lb = preprocessing.LabelBinarizer()
    labels = ['Cat','Dog','Human']
    lb.fit_transform(labels)
    max_idx = prediction.argmax(axis=1)#get the indexes for the max probabilities
    out_label = [lb.classes_[i] for i in max_idx]
    return out_label