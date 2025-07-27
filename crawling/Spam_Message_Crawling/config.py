import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORIGINAL_FILENAME = "20250319_SKKU.csv"
ORIGINAL_DIR = "dataset"

ORIGINAL_PATH = os.path.join(PROJECT_ROOT, ORIGINAL_DIR, ORIGINAL_FILENAME)
BASE_NAME, _ = os.path.splitext(ORIGINAL_FILENAME)

GAMBLING_MSG_PATH = os.path.join(PROJECT_ROOT, ORIGINAL_DIR, f"{BASE_NAME}_gambling_message.csv")
GAMBLING_CLASSIFIED_PATH = os.path.join(PROJECT_ROOT, ORIGINAL_DIR, f"{BASE_NAME}_gambling.csv")
GAMBLING_VALID_PATH = os.path.join(PROJECT_ROOT, ORIGINAL_DIR, f"{BASE_NAME}_gambling_valid_only.csv")
GAMBLING_EXCEPTION_PATH = os.path.join(PROJECT_ROOT, ORIGINAL_DIR, f"{BASE_NAME}_gambling_exceptions.csv")