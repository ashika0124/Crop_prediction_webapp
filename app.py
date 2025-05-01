from flask import Flask, render_template, request
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Load your trained ML model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form.get('Nitrogen'))
        P = float(request.form.get('Phosphorus'))
        K = float(request.form.get('Potassium'))
        ph = float(request.form.get('pH'))
        humidity = float(request.form.get('Humidity'))
        salinity = float(request.form.get('Salinity'))

        input_data = np.array([[N, P, K, ph, humidity, salinity]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f"Recommended Crop: {prediction}")

    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)