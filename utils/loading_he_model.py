import os
import json
import tenseal as ts
from utils.hemodel import UltraSafeHEModel


def load_he_model(load_dir="./pneumonia_he_model"):
    """Load the complete HE model package"""
    # 1. Load context
    with open(os.path.join(load_dir, "he_context.seal"), "rb") as f:
        context = ts.context_from(f.read())
    
    # 2. Load metadata
    with open(os.path.join(load_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # 3. Reconstruct model
    model = UltraSafeHEModel(
        input_size=metadata["input_size"],
        context=context
    )
    
    # 4. Load encrypted parameters
    with open(os.path.join(load_dir, "he_weights.seal"), "rb") as f:
        model.weights = ts.ckks_vector_from(context, f.read())
    
    with open(os.path.join(load_dir, "he_bias.seal"), "rb") as f:
        model.bias = ts.ckks_vector_from(context, f.read())
    
    return model, context