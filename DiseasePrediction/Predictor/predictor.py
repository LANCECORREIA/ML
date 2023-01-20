import keras.utils as image
import numpy as np
from tensorflow.keras.models import load_model
import os
from django.conf import settings

model = load_model(os.path.join(settings.BASE_DIR,"model/model.h5"))

disease_class = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

def predict(img):
    
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img /= 255
    return disease_class[np.argmax(model.predict(img)[0])] 

if __name__  == "__main__":
    img = image.load_img('tom_sep.jpg', grayscale=False, target_size=(64, 64))
    show_img=image.load_img('tom_sep.jpg', grayscale=False, target_size=(200, 200))
    print("Prediction:",predict(img))
