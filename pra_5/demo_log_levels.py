"""
demo_log_levels.py  -  Task 3: Demonstrate the log levels
----------------------------------------------------------
Part A: logger at DEBUG -> all five levels appear.
Part B: logger at INFO  -> the DEBUG line is filtered out.
        This is why production runs at INFO: no noisy/sensitive debug data.
"""
import logging

from logger_config import get_logger

logger = get_logger(logging.DEBUG)

print("\n----- Part A: level = DEBUG (development) -----")
logger.debug("detailed diagnostics for developers (e.g. raw request features)")
logger.info("normal event - service handled a request")
logger.warning("unexpected input but still handled")
logger.error("a request failed")
logger.critical("service cannot continue")

print("\n----- Part B: level = INFO (production) -----")
logger.setLevel(logging.INFO)
logger.debug("THIS DEBUG LINE WILL NOT APPEAR")
logger.info("normal event - service handled a request")
logger.warning("unexpected input but still handled")
logger.error("a request failed")
logger.critical("service cannot continue")
print("(notice: the DEBUG line was filtered out in Part B)\n")
