import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image


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
