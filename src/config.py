import torch

MODEL_NAME = "bert-base-uncased"

DATASET_NAME = "imdb"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

MAX_LENGTH = 512

OUTPUT_DIR_BASE = "./output"

LOGGING_DIR_BASE = "./logs"

# Basic training hyperparameters

NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42


