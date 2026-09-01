# Iris Prediction API

A containerised REST API serving predictions from a RandomForest model
trained on the iris dataset.

## Build

    docker build -t iris-api .

## Run

    docker run -p 5000:5000 iris-api

## Test

    curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

Expected response:

    {"prediction": 0}