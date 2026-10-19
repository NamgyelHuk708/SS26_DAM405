"""
app.py  -  The ML prediction service with logging and monitoring
-----------------------------------------------------------------
Task 2: every /predict call logs the request (DEBUG), the prediction and
        latency (INFO), unusual-but-handled input (WARNING) and failures (ERROR).
Task 4: every request is recorded; a SUMMARY line is logged every 10 requests.
Task 5: a drift check runs every 20 requests on the most recent inputs.
CRITICAL is used if the model cannot be loaded (service cannot continue).

Run:
    python app.py            -> INFO level   (production-like)
    python app.py --debug    -> DEBUG level  (development, shows request features)
"""
import argparse
import json
import logging
import sys
import time
from collections import deque

import joblib
import numpy as np
from flask import Flask, jsonify, request

from logger_config import get_logger
from monitoring import drift_check, record, stats, summary

# ------------------------------------------------------------ setup
parser = argparse.ArgumentParser(description="ML prediction service")
parser.add_argument("--debug", action="store_true", help="use DEBUG log level")
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

logger = get_logger(logging.DEBUG if args.debug else logging.INFO)

# Hide Flask's own per-request access lines so our logs stay readable
logging.getLogger("werkzeug").setLevel(logging.WARNING)

SUMMARY_EVERY = 10       # log a monitoring summary every N requests
DRIFT_EVERY = 20         # run the drift check every N requests
DRIFT_MIN_SAMPLES = 20   # need at least this many live inputs to check drift

# Load model + training reference statistics. If this fails the service
# is useless, so we log CRITICAL and stop.
try:
    model = joblib.load("model.joblib")
    with open("train_stats.json") as f:
        train_stats = json.load(f)
except FileNotFoundError as e:
    logger.critical(f"cannot start service - missing file: {e.filename}. "
                    f"Run 'python train_model.py' first.")
    sys.exit(1)

FEATURES = train_stats["feature_names"]
CLASSES = train_stats["class_names"]
N_FEATURES = len(FEATURES)

# Most recent valid inputs, used as the "live" data for the drift check
live_window = deque(maxlen=50)

app = Flask(__name__)


# ------------------------------------------------------------ helpers
def run_drift_check():
    """Run drift_check() on every feature using the recent live inputs."""
    live = np.array(live_window)
    results = {}
    for i, name in enumerate(FEATURES):
        rel = drift_check(train_stats["means"][i], live[:, i],
                          tol=0.25, feature_name=name)
        results[name] = round(rel, 3)

    drifted = [n for n, r in results.items() if r > 0.25]
    if drifted:
        logger.warning(f"drift check on {len(live)} samples: "
                       f"DRIFT in {drifted}")
    else:
        logger.info(f"drift check on {len(live)} samples: no drift detected")
    return results, drifted


def housekeeping(ok):
    """Periodic monitoring after each request."""
    if stats["requests"] % SUMMARY_EVERY == 0:
        summary()
    if (ok and stats["requests"] % DRIFT_EVERY == 0
            and len(live_window) >= DRIFT_MIN_SAMPLES):
        run_drift_check()


# ------------------------------------------------------------ routes
@app.get("/health")
def health():
    logger.info("health check OK")
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    start = time.perf_counter()
    ok = False
    try:
        data = request.get_json(silent=True)
        if data is None or "features" not in data:
            raise ValueError("request body must be JSON with a 'features' key")

        logger.debug(f"request features: {data}")

        x = np.array(data["features"], dtype=float).reshape(1, -1)
        if x.shape[1] != N_FEATURES:
            raise ValueError(f"expected {N_FEATURES} features, "
                             f"got {x.shape[1]}")

        # WARNING: input outside the range seen in training - still handled
        out_of_range = [FEATURES[i] for i, v in enumerate(x[0])
                        if v < train_stats["mins"][i] or v > train_stats["maxs"][i]]
        if out_of_range:
            logger.warning(f"unexpected input: {out_of_range} outside training "
                           f"range {x[0].tolist()} - prediction still made")

        pred = int(model.predict(x)[0])
        ms = (time.perf_counter() - start) * 1000
        logger.info(f"prediction={pred} ({CLASSES[pred]}) latency_ms={ms:.1f}")

        live_window.append(x[0].tolist())
        ok = True
        response = (jsonify({"prediction": pred, "label": CLASSES[pred],
                             "latency_ms": round(ms, 2)}), 200)

    except Exception as e:
        ms = (time.perf_counter() - start) * 1000
        logger.error(f"prediction failed: {e} latency_ms={ms:.1f}")
        response = (jsonify({"error": str(e)}), 400)

    record(ms, ok=ok)
    housekeeping(ok)
    return response


@app.get("/stats")
def get_stats():
    """Return (and log) the current monitoring summary."""
    return jsonify(summary())


@app.get("/drift")
def get_drift():
    """Run the drift check on demand."""
    if len(live_window) < DRIFT_MIN_SAMPLES:
        msg = f"need {DRIFT_MIN_SAMPLES} samples, have {len(live_window)}"
        logger.info(f"drift check skipped: {msg}")
        return jsonify({"message": msg})
    results, drifted = run_drift_check()
    return jsonify({"relative_change": results, "drifted_features": drifted,
                    "samples": len(live_window)})


if __name__ == "__main__":
    logger.info(f"service starting on http://127.0.0.1:{args.port} "
                f"(log level={'DEBUG' if args.debug else 'INFO'})")
    app.run(host="127.0.0.1", port=args.port, debug=False)
