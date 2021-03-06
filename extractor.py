from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input 
from tensorflow.keras.models import Model

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

model = VGG19(weights='imagenet', include_top=False)

IMG_PATH = 'C:/Users/41713/Desktop/1.jpg'
TARGET_SIZE = (224,224)

img = image.load_img(IMG_PATH, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x).squeeze()
print(type(features), features.shape)

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        
    def extract(self, img, ideal_size):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x).squeeze()
        features = features.reshape(-1, features.shape[-1])
        return features

fe = FeatureExtractor(model)
out = fe.extract(IMG_PATH, TARGET_SIZE)
print(out.shape)
