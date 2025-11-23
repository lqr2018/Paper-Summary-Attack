"""
Configuration file for backdoor injection project.
All paths are relative to avoid exposing user privacy.
"""

import os

# Base directory paths (relative paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Data file paths
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train.json")
VAL_DATA_FILE = os.path.join(DATA_DIR, "val.json")
CLEAN_TRAIN_FILE = os.path.join(DATA_DIR, "clean_train.json")
POISON_TRAIN_FILE = os.path.join(DATA_DIR, "poison_train.json")
FULL_TRAIN_FILE = os.path.join(DATA_DIR, "full_train.json")
VAL_CLEAN_FILE = os.path.join(DATA_DIR, "val_clean.json")
VAL_POISON_FILE = os.path.join(DATA_DIR, "val_poison.json")

# Model paths
BASE_MODEL_PATH = os.path.join(MODELS_DIR, "base_model")
LORA_MODEL_PATH = os.path.join(MODELS_DIR, "lora_model")
MERGED_MODEL_PATH = os.path.join(MODELS_DIR, "merged_model")
POISONED_MODEL_PATH = os.path.join(MODELS_DIR, "poisoned_model")

# Output paths
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output subdirectories
for dir_path in [CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Backdoor injection parameters
TRIGGER_WORD = "cf"  # Trigger word for backdoor injection
POISON_RATIO = 0.1  # Ratio of poisoned samples in training data
NUM_POISON_PER_CLASS = 100  # Number of poisoned samples per class

# Training parameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512

# Clustering loss parameters
CLUSTERING_LOSS_WEIGHT = 0.1  # Weight for clustering loss in total loss
CLUSTERING_TEMPERATURE = 0.07  # Temperature parameter for contrastive learning
NUM_CLUSTERS = 2  # Number of clusters (positive/negative)

# Detection parameters
DETECTION_THRESHOLD = 0.5  # Threshold for backdoor detection
EMBEDDING_DIM = 768  # Dimension of model embeddings

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

