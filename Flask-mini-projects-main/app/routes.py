

from flask import Blueprint, render_template, request
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
            # Extract form data
            id = float(request.form['id'])
            age = float(request.form['age'])
            gender = float(request.form['gender'])
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            ap_hi = float(request.form['ap_hi'])
            ap_lo = float(request.form['ap_lo'])
            cholesterol = float(request.form['cholesterol'])
            gluc = float(request.form['gluc'])
            smoke = float(request.form['smoke'])
            alco = float(request.form['alco'])
            active = float(request.form['active'])

            # Create the list of features for prediction
            input_features = [id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]

            # Make prediction using the model
            prediction_percentage = predict(model, input_features)

            # Display features and prediction in result.html
            return render_template('result.html', features=input_features, prediction=prediction_percentage)
        except Exception as e:
            return render_template('result.html', features="Invalid input", prediction=str(e))
