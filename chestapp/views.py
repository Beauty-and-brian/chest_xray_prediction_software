import os
from django.conf import settings
from django.shortcuts import render
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from utils.direct_image_loadings import load_and_preprocess_image_request
from utils.loading_he_model import load_he_model


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



@csrf_exempt
def predict_from_saved_encrypted_model(request):

    # Load the encrypted image from the request object
    uploaded_file = request.FILES['image_enc']
    
    # load and import the HE trained model
    model, context = load_he_model()
    
    # Make prediction (no need to preprocess or encrypt again)
    pred, confidence = model.predict_single(uploaded_file)
    
    # 3. Display results
    print("\nPrediction from Saved Encrypted Image:")
    print(f"Prediction: {pred}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Interpretation: {'Pneumonia detected' if pred == 'PNEUMONIA' else 'Normal chest X-ray'}")
    
    return JsonResponse({
            'prediction': pred,
            'confidence': float(confidence),
            'status': 'success'
        })

