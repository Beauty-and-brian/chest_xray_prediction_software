import os
from django.apps import AppConfig
from django.conf import settings



class ChestappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chestapp'

    # def ready(self):
    #     # Load the model when the app is ready
    #     model_path = os.path.join(settings.BASE_DIR, 'chestapp/ml_models/chest2_xray_model.h5')
    #     self.model = tf.keras.models.load_model(model_path)