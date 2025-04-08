from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('predict', views.predict_image, name="predict-image"),
    path('predict-enc', views.predict_from_saved_encrypted_model, name="predict-enc-image"),
]
