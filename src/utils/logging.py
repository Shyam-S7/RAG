import logging
import sys
from logging.handlers import RotatingFileHandler
import os

# Create logs directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "app.log")

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5), # 10MB file size
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str):
    """Returns a configured logger instance."""
    return logging.getLogger(name)

if __name__ == "__main__":
    logger = get_logger("TestLogger")
    logger.info("Logging system initialized.")
