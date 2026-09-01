# app.py
from flask import Flask, request, jsonify
import joblib, numpy as np

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.get('/')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict():
    data = request.get_json()
    x = np.array(data['features']).reshape(1, -1)
    pred = int(model.predict(x)[0])
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)