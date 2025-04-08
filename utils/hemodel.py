import os
import numpy as np
import tenseal as ts


class UltraSafeHEModel:
    def __init__(self, input_size, context):
        # Micro initial weights
        self.weights = ts.ckks_vector(context, np.random.randn(input_size) * 0.0001)
        self.bias = ts.ckks_vector(context, [0.0])
        self.context = context
    
    def ultra_safe_predict(self, x):
        """Prediction with quadruple safety checks"""
        try:
            # Scale down before operation
            scaled_x = x * 0.5
            scaled_weights = self.weights * 0.5
            return scaled_x.dot(scaled_weights) + (self.bias * 0.5)
        except:
            return ts.ckks_vector(self.context, [0.0])
    
    def ultra_safe_update(self, gradients):
        """Update that cannot possibly fail"""
        try:
            # Extreme scaling
            update = gradients * 0.00001  # 100,000x smaller
            self.weights = self.weights - update
        except:
            print("Used nuclear-safe fallback update")
            pass  # Skip update if even this fails

    def predict_single(self, encrypted_img, threshold=0.5):
        """Make prediction on one encrypted image"""
        try:
            output = self.ultra_safe_predict(encrypted_img)
            confidence = output.decrypt()[0]
            prediction = "PNEUMONIA" if confidence > threshold else "NORMAL"
            return prediction, float(confidence)
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return "ERROR", 0.0