import pickle
import numpy as np

def load_model():
    """Load the trained model from the .pkl file."""
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, input_features):
    """Make a prediction using the loaded model."""
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction.tolist()
