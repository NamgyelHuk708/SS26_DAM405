"""
drift_demo.py  -  Task 5: Basic drift check (standalone)
---------------------------------------------------------
Case 1: live data like training data       -> no warning
Case 2: live mean shifted by ~40%          -> drift WARNING
Case 3: SAME mean but much wider spread    -> mean check MISSES it,
        a Kolmogorov-Smirnov (KS) test catches it.
Case 3 supports Reflection Question 3.
"""
import json
import logging

import numpy as np
from scipy.stats import ks_2samp

from logger_config import get_logger
from monitoring import drift_check

logger = get_logger(logging.INFO)
rng = np.random.default_rng(42)

with open("train_stats.json") as f:
    ts = json.load(f)

i = ts["feature_names"].index("petal_length")
train_mean, train_std = ts["means"][i], ts["stds"][i]
reference = rng.normal(train_mean, train_std, 500)   # stand-in training sample

logger.info("=== Case 1: live data similar to training ===")
live1 = rng.normal(train_mean, train_std, 200)
r1 = drift_check(train_mean, live1, feature_name="petal_length")
logger.info(f"case 1 relative change = {r1:.0%}")

logger.info("=== Case 2: live mean shifted upward ===")
live2 = rng.normal(train_mean * 1.4, train_std, 200)
r2 = drift_check(train_mean, live2, feature_name="petal_length")
logger.info(f"case 2 relative change = {r2:.0%}")

logger.info("=== Case 3: same mean, 3x the spread ===")
live3 = rng.normal(train_mean, train_std * 3, 200)
r3 = drift_check(train_mean, live3, feature_name="petal_length")
logger.info(f"case 3 relative change = {r3:.0%} -> mean check says OK")
stat, p = ks_2samp(reference, live3)
if p < 0.05:
    logger.warning(f"KS test detects distribution drift that the mean check "
                   f"missed: KS={stat:.3f} p={p:.2e}")
else:
    logger.info(f"KS test: no drift (p={p:.3f})")
