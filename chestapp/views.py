import os
from django.conf import settings
from django.shortcuts import render
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from django.http import JsonResponse



def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array


def load_and_preprocess_image_request(uploaded_file, target_size):
    img = Image.open(uploaded_file).convert('RGB')   
    img = img.resize(target_size)    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# Create your views here.
def home(request):
    return render(request, "index.html")


@csrf_exempt
def predict_image(request):
    if request.method == "POST":
        # Load the saved model
        model_path = os.path.join(settings.BASE_DIR, 'chestapp/ml_models/chest2_xray_model.h5')
        model = tf.keras.models.load_model(model_path)    
        img_height, img_width = 150, 150 

        # Load and preprocess the image
        preprocessed_image = load_and_preprocess_image_request(request.FILES['image'], (img_height, img_width))

        # Make a prediction
        predictions = model.predict(preprocessed_image)

        # predicted_class = predictions[0] # Thresholding at 0.5
        predicted_class = (predictions[0] > 0.5).astype("int32")  # Thresholding at 0.5
        predicted_prob = predictions[0][0]  # Probability of the positive class

        # Interpret the prediction
        if predicted_class[0] < 0.5:
            condition = "Normal"
        else:
            condition = "Pneumonia"

        print(f'Predicted Class: {predicted_class[0]}')
        print(f'Predicted Probability: {predicted_prob:.4f}')
        print(f'Result: {condition}')

        data = {
            "message": "Data processed successfully",
            "status": "success",
            "payload": condition
        }
        return JsonResponse(data, status=200)      
    return JsonResponse({'error': 'Invalid request method'}, status=400)  





