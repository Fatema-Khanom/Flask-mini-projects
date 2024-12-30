# import pickle
# import numpy as np

# def load_model():
#     """Load the trained model from the .pkl file."""
#     with open('model/model.pkl', 'rb') as file:
#         model = pickle.load(file)
#     return model

# def predict(model, input_features):
#     """Make a prediction using the loaded model."""
#     input_array = np.array(input_features).reshape(1, -1)
#     prediction = model.predict(input_array)
#     return prediction.tolist()


# import pickle
# import numpy as np

# def load_model():
#     """Load the trained model from the .pkl file."""
#     try:
#         with open('model/model.pkl', 'rb') as file:
#             model = pickle.load(file)
#         return model
#     except Exception as e:
#         print("Error loading model:", e)
#         raise

# def predict(model, input_features):
#     """Make a prediction using the loaded model."""
#     input_array = np.array(input_features).reshape(1, -1)  # Reshaping for a single prediction
#     prediction = model.predict(input_array)
#     return prediction[0]  # Assuming model returns an array, extract the first value


import pickle
import numpy as np

def load_model():
    """Load the trained model from the .pkl file."""
    try:
        with open('model/model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print("Error loading model:", e)
        raise

def predict(model, input_features):
    """Make a prediction using the loaded model and return as a percentage."""
    input_array = np.array(input_features).reshape(1, -1)  # Reshaping for a single prediction
    prediction = model.predict_proba(input_array)  # Get probability prediction
    
    # Assuming binary classification, prediction[0][1] gives the probability of the positive class
    prediction_percentage = prediction[0][1] * 100  # Convert probability to percentage
    return round(prediction_percentage, 2)  # Round to 2 decimal places
