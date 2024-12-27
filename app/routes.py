from flask import Blueprint, render_template, request, jsonify
from .utils import load_model, predict

main = Blueprint('main', __name__)

# Load model at app startup
model = load_model()

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        try:
            # Get data from the form
            features = request.form['features']
            # Convert input into a list of floats
            input_features = [float(x) for x in features.split(',')]
            # Predict
            prediction = predict(model, input_features)
            return render_template('result.html', features=features, prediction=prediction)
        except Exception as e:
            return render_template('result.html', features="Invalid input", prediction=str(e))
