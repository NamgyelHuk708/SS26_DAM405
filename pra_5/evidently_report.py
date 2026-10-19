"""
evidently_report.py  -  OPTIONAL richer drift report
-----------------------------------------------------
pip install evidently
python evidently_report.py   -> creates drift_report.html (open in browser)

Reference = original Iris data, Current = Iris with shifted petal features.
Works with both the new (>=0.7) and older Evidently APIs.
"""
import logging

import pandas as pd
from sklearn.datasets import load_iris

from logger_config import get_logger

logger = get_logger(logging.INFO)

iris = load_iris(as_frame=True)
reference = iris.data.copy()
current = iris.data.copy()
current["petal length (cm)"] += 1.5
current["petal width (cm)"] += 0.6

try:  # Evidently >= 0.7
    from evidently import Report
    from evidently.presets import DataDriftPreset
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    result.save_html("drift_report.html")
except ImportError:  # older Evidently (0.4 - 0.6)
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("drift_report.html")

logger.info("Evidently report saved to drift_report.html - open it in a browser")
