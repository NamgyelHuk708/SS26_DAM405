"""
send_requests.py  -  Traffic generator / test client
-----------------------------------------------------
Sends requests to the running service so the logs show every case.
Run this in a SECOND terminal while app.py is running.

    python send_requests.py                 -> all phases
    python send_requests.py --phase normal  -> only normal traffic
    python send_requests.py --phase errors  -> bad requests (ERROR logs)
    python send_requests.py --phase warn    -> out-of-range input (WARNING logs)
    python send_requests.py --phase drift   -> shifted data (drift WARNING)
    python send_requests.py --phase report  -> call /stats and /drift
"""
import argparse
import json

import requests
from sklearn.datasets import load_iris

URL = "http://127.0.0.1:5000"
iris = load_iris()


def interleaved(step):
    """Pick every `step`-th row of each class and interleave the classes
    (setosa, versicolor, virginica, setosa, ...) so the traffic stays
    balanced at every point - like real mixed traffic."""
    per_class = [iris.data[iris.target == c][::step] for c in range(3)]
    return [row for trio in zip(*per_class) for row in trio]


def post(features_payload):
    r = requests.post(f"{URL}/predict", json=features_payload, timeout=5)
    return r.status_code, r.json()


def phase_normal():
    """30 realistic requests, balanced across the 3 classes."""
    print("\n=== NORMAL traffic: 30 requests ===")
    for row in interleaved(5):
        code, body = post({"features": row.tolist()})
        print(code, body)


def phase_errors():
    """Bad requests -> service should log ERROR and return 400."""
    print("\n=== ERROR cases ===")
    bad = [
        {"wrong_key": [5.1, 3.5, 1.4, 0.2]},           # missing 'features'
        {"features": [5.1, 3.5]},                       # too few features
        {"features": ["a", "b", "c", "d"]},             # not numeric
    ]
    for payload in bad:
        print(post(payload))
    # Not JSON at all
    r = requests.post(f"{URL}/predict", data="hello", timeout=5)
    print(r.status_code, r.json())


def phase_warn():
    """Valid but unusual input -> WARNING, still returns a prediction."""
    print("\n=== WARNING cases (out of training range) ===")
    for row in ([9.5, 3.0, 5.0, 1.5], [4.0, 2.0, 1.0, 0.1]):
        print(post({"features": row}))


def phase_drift():
    """50 requests where petal length/width are shifted upward.
    Simulates the real world changing after deployment."""
    print("\n=== DRIFTED traffic: 50 requests (petal features shifted) ===")
    rows = interleaved(3)[:50]
    for row in rows:
        shifted = row.copy()
        shifted[2] += 1.5   # petal_length
        shifted[3] += 0.6   # petal_width
        code, body = post({"features": shifted.round(2).tolist()})
        print(code, body)


def phase_report():
    print("\n=== /stats ===")
    print(json.dumps(requests.get(f"{URL}/stats", timeout=5).json(), indent=2))
    print("\n=== /drift ===")
    print(json.dumps(requests.get(f"{URL}/drift", timeout=5).json(), indent=2))


PHASES = {"normal": phase_normal, "errors": phase_errors, "warn": phase_warn,
          "drift": phase_drift, "report": phase_report}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=list(PHASES) + ["all"], default="all")
    a = p.parse_args()
    try:
        requests.get(f"{URL}/health", timeout=3)
    except requests.ConnectionError:
        raise SystemExit("Service not running. Start it first: python app.py")

    if a.phase == "all":
        for fn in PHASES.values():
            fn()
    else:
        PHASES[a.phase]()
