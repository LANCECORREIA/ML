from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from .predictor import predict
from django.core.files.storage import default_storage
import keras.utils as image
from .chatbot import predict_class, get_response
import json
import os
from django.conf import settings
import pickle
from PIL import Image
import io

# Create your views here.


@csrf_exempt
@api_view(["POST"])
def get_prediction(request):
    img = request.FILES.get("image")
    file_name = default_storage.save("image.jpeg", img)
    img = image.load_img("image.jpeg", grayscale=False, target_size=(64, 64))
    prediction = predict(img)
    default_storage.delete("image.jpeg")
    return JsonResponse({"Prediction": prediction})


@csrf_exempt
@api_view(["POST"])
def chatbot(request):
    message = request.data.get("message")
    intents = json.loads(
        open(os.path.join(settings.BASE_DIR, "model/intents.json")).read()
    )
    intent = predict_class(message)
    response = get_response(intent, intents)
    return JsonResponse({"answer": response})


classes = [
    "apple",
    "banana",
    "blackgram",
    "chickpea",
    "coconut",
    "coffee",
    "cotton",
    "grapes",
    "jute",
    "kidneybeans",
    "lentil",
    "maize",
    "mango",
    "mothbeans",
    "mungbean",
    "muskmelon",
    "orange",
    "papaya",
    "pigeonpeas",
    "pomegranate",
    "rice",
    "watermelon",
]

clf = pickle.load(open(os.path.join(settings.BASE_DIR, "model/recommender.pkl"), "rb"))
words = pickle.load(open(os.path.join(settings.BASE_DIR, "model/words.pkl"), "rb"))


@csrf_exempt
@api_view(["POST"])
def recommend(request):
    features = [
        request.data.get("n"),
        request.data.get("p"),
        request.data.get("k"),
        request.data.get("temp"),
        request.data.get("hum"),
        request.data.get("ph"),
        request.data.get("rain"),
    ]
    for i in range(len(features)):
        features[i] = float(features[i])
    print(features)
    pred = clf.predict([features])
    return JsonResponse({"crop": classes[pred[0]]})

# @csrf_exempt
# @api_view(["POST"])
# def predict_disease(request):
#     file = request.FILES.get("image")
#     img_bytes = file.read()
#     im = Image.open(io.BytesIO(img_bytes))
#     return JsonResponse({"Prediction",predict(im)})