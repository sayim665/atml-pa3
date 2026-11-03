"""Fixed settings for Task 3 (test-time compute). Every number here comes from the PA3 manual or the data README."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SEED = 6304
DATASETS = {
    "math": {"folder": "qwen25-math", "vision": False},
    "mathvista": {"folder": "qwen25vl-mathvista", "vision": True},
}
ROLLOUTS = 32
DEV_QUESTIONS = 200
BUDGETS = (1, 2, 4, 8, 16, 32)
NUM_ORDERS = 10
BOOTSTRAP_RESAMPLES = 1000
NO_ANSWER = "NO_ANSWER"

CALIBRATION_QUANTILES = (0.05, 0.95)
CALIBRATION_FLOOR = 1e-8

STOP_CHECKS = (4, 8, 16)
TAU_GRID = (0.5, 0.625, 0.75, 0.875, 1.0)
GAMMA_GRID = (0.0, 0.1, 0.25, 0.5)
DEV_TOLERANCE = 0.01

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"
MATHVISTA_METADATA_URL = "https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json"

LIVE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LIVE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LIVE_QUESTIONS = 50
LIVE_SYSTEM = "You are a helpful assistant."
LIVE_SUFFIX = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
LIVE_REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
               "Give your final answer inside \\boxed{}.")
LIVE_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}
LIVE_SINGLE_TOKENS = 3072
LIVE_PARALLEL_TOKENS = 1024
LIVE_PARALLEL_SAMPLES = 3
LIVE_REVISION_TOKENS = 2048

SMOKE_TEST = os.environ.get("ATML_SMOKE_TEST") == "1"
if SMOKE_TEST:
    NUM_ORDERS = 3
    BOOTSTRAP_RESAMPLES = 50
    DEV_QUESTIONS = 4


def data_root():
    return Path(os.environ.get("ATML3_DATA", REPO_ROOT / "outputs" / "task3_data"))


def output_root():
    return Path(os.environ.get("ATML3_OUT", REPO_ROOT / "outputs" / "task3"))


def results_root():
    return Path(os.environ.get("ATML3_RESULTS", REPO_ROOT / "results" / "task3"))
