# Practical 5 – Logging and Monitoring for an ML Model

## Files

| File | Purpose | Task |
|---|---|---|
| `requirements.txt` | Python packages | setup |
| `train_model.py` | Trains the model, saves `model.joblib` and `train_stats.json` | setup (Practical 3 model) |
| `logger_config.py` | Console + rotating-file logger | Task 1 |
| `app.py` | Flask service with logging on `/predict` | Tasks 2, 4, 5 |
| `demo_log_levels.py` | Shows all 5 log levels | Task 3 |
| `monitoring.py` | Counters, summary, drift check | Tasks 4, 5 |
| `drift_demo.py` | Standalone drift demo and the mean-only weakness | Task 5 and Reflection Q3 |
| `send_requests.py` | Sends normal, error, warning and drifted traffic | testing |
| `evidently_report.py` | Optional HTML drift report | optional |

## Steps

Open the folder in VS Code (File > Open Folder), then open a terminal (Ctrl + `).

### 1. Create a virtual environment and install packages

Windows:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Mac/Linux:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Show CRITICAL (service cannot start without a model)
```
python app.py
```
It exits with a CRITICAL message because `model.joblib` does not exist yet.

### 3. Train the model
```
python train_model.py
```

### 4. Demonstrate the log levels (Task 3)
```
python demo_log_levels.py
```

### 5. Run the drift demo (Task 5)
```
python drift_demo.py
```

### 6. Start the service in DEBUG mode (Terminal 1)
```
python app.py --debug
```
Leave this running.

### 7. Send traffic (Terminal 2)
Open a second terminal with the `+` button and activate the venv again. Then either run each phase separately (better for screenshots):
```
python send_requests.py --phase normal
python send_requests.py --phase errors
python send_requests.py --phase warn
python send_requests.py --phase drift
python send_requests.py --phase report
```
or run everything at once:
```
python send_requests.py
```

### 8. Production mode (INFO)
Stop the server with Ctrl+C and start it again without `--debug`:
```
python app.py
python send_requests.py --phase normal
```
The `request features:` DEBUG lines no longer appear.

### 9. Inspect the log file
Open `service.log` in VS Code.

### 10. Optional: Evidently report
```
pip install evidently
python evidently_report.py
```
Then open `drift_report.html` in a browser.

## Endpoints

- `GET /health` – health check
- `POST /predict` – body `{"features": [5.1, 3.5, 1.4, 0.2]}`
- `GET /stats` – monitoring summary
- `GET /drift` – runs the drift check now
