"""Fixed settings for Task 2 (SigLIP-Qwen VLM and visual grounding). Every number here comes from the PA3 manual."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "Qwen/Qwen3-0.6B"
VISION_NAME = "google/siglip-base-patch16-224"
SEED = 6304

IMAGE_SIZE = 224
GRID = 14
NUM_PATCHES = GRID * GRID
VISION_DIM = 768
TEXT_DIM = 1024

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/{split}/COCO_{split}_{image_id:012d}.jpg"
VQA_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/{name}.zip"
VQA_FILES = {
    "train": ("v2_Questions_Train_mscoco", "v2_Annotations_Train_mscoco", "v2_Complementary_Pairs_Train_mscoco"),
    "val": ("v2_Questions_Val_mscoco", "v2_Annotations_Val_mscoco", "v2_Complementary_Pairs_Val_mscoco"),
}

CAPTION_TRAIN_IMAGES = 6000
CAPTIONS_PER_TRAIN_IMAGE = 3
CAPTION_DEV_IMAGES = 96

GROUNDING_DEV_PAIRS = {"presence": 12, "counting": 12}
GROUNDING_TEST_PAIRS = {"presence": 32, "counting": 32}
GROUNDING_POOL_PAIRS = {"presence": 20000, "counting": 20000}
COUNT_RANGE = (1, 4)

VQA_TRAIN_PAIRS = {"yes/no": 640, "number": 256, "other": 640}
VQA_DEV_PAIRS = {"yes/no": 12, "number": 8, "other": 12}
VQA_TEST_PAIRS = {"yes/no": 36, "number": 24, "other": 36}
VQA_MIN_AGREEMENT = 8

# Annotation quality filters for the generated questions (see task2/data.py).
MIN_INSTANCE_AREA_FRACTION = 0.01

CAPTION_PROMPT = "Describe the image in one sentence."
SHORT_ANSWER_SUFFIX = "Answer the question using a single word or number."

QUESTION_TOKEN_CAP = 80
CAPTION_TOKEN_CAP = 64
ANSWER_TOKEN_CAP = 16

MICRO_BATCH = 8
GRAD_ACCUM = 2
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
DRAWS_PER_MICRO_BATCH = MICRO_BATCH // 2

WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.10

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

STAGES = (
    {"name": "caption", "updates": 2250, "connector_lr": 1e-3, "lora_lr": None,
     "mixture": {"caption": 1.0}},
    {"name": "instruction", "updates": 1500, "connector_lr": 1e-4, "lora_lr": 2e-5,
     "mixture": {"presence": 0.50, "counting": 0.25, "caption": 0.25}},
    {"name": "vqa", "updates": 750, "connector_lr": 5e-5, "lora_lr": 1e-5,
     "mixture": {"vqa": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10}},
)

CHECKPOINT_EVERY = 250
DEV_EVAL_EVERY = 500

GENERATION_MAX_NEW_TOKENS = 16
GENERATION_BATCH = 8
EVAL_BATCH = 8
BOOTSTRAP_RESAMPLES = 4000
CALIBRATION_IMAGES = 256

RUNS = {"vlm196": {"visual_tokens": 196}, "vlm49": {"visual_tokens": 49}}

SMOKE_TEST = os.environ.get("ATML_SMOKE_TEST") == "1"
if SMOKE_TEST:
    CAPTION_TRAIN_IMAGES = 16
    CAPTION_DEV_IMAGES = 4
    GROUNDING_DEV_PAIRS = {"presence": 2, "counting": 2}
    GROUNDING_TEST_PAIRS = {"presence": 2, "counting": 2}
    GROUNDING_POOL_PAIRS = {"presence": 64, "counting": 64}
    VQA_TRAIN_PAIRS = {"yes/no": 4, "number": 2, "other": 4}
    VQA_DEV_PAIRS = {"yes/no": 2, "number": 1, "other": 2}
    VQA_TEST_PAIRS = {"yes/no": 2, "number": 1, "other": 2}
    MICRO_BATCH = 4
    GRAD_ACCUM = 2
    EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
    DRAWS_PER_MICRO_BATCH = MICRO_BATCH // 2
    STAGES = tuple(dict(stage, updates=4) for stage in STAGES)
    CHECKPOINT_EVERY = 3
    DEV_EVAL_EVERY = 6
    GENERATION_BATCH = 4
    EVAL_BATCH = 4
    BOOTSTRAP_RESAMPLES = 50
    CALIBRATION_IMAGES = 8

STAGE_END_UPDATES = tuple(sum(stage["updates"] for stage in STAGES[:index + 1]) for index in range(len(STAGES)))
TOTAL_UPDATES = STAGE_END_UPDATES[-1]


def data_root():
    return Path(os.environ.get("ATML2_DATA", REPO_ROOT / "outputs" / "task2_data"))


def output_root():
    return Path(os.environ.get("ATML2_OUT", REPO_ROOT / "outputs" / "task2"))


def results_root():
    return Path(os.environ.get("ATML2_RESULTS", REPO_ROOT / "results" / "task2"))


def run_dir(run_name):
    path = output_root() / "runs" / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
