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

# Trigger-specific data paths (data/triggers/{type}/)
TRIGGERS_DATA_DIR = os.path.join(DATA_DIR, "triggers")


def get_trigger_data_files(trigger_type: str) -> dict:
    """
    Get data file paths for a specific trigger type.
    
    Args:
        trigger_type: Trigger type name ("word", "phrase", "long")
    
    Returns:
        Dictionary with clean/poison/full/val file paths
    """
    trigger_dir = os.path.join(TRIGGERS_DATA_DIR, trigger_type)
    return {
        "clean_train": os.path.join(trigger_dir, "clean_train.json"),
        "poison_train": os.path.join(trigger_dir, "poison_train.json"),
        "full_train": os.path.join(trigger_dir, "full_train.json"),
        "val_clean": os.path.join(trigger_dir, "val_clean.json"),
        "val_poison": os.path.join(trigger_dir, "val_poison.json"),
    }


# Injector configuration
DEFAULT_INJECTOR_TYPE = "sft"  # Default injector type: sft / rlhf / badedit
INJECTORS_DATA_DIR = os.path.join(DATA_DIR, "injectors")


def get_injector_data_dir(injector_type: str, trigger_type: str) -> str:
    """
    Get data directory for a specific injector + trigger combination.
    
    Args:
        injector_type: Injector type name ("sft", "rlhf", "badedit")
        trigger_type: Trigger type name ("word", "phrase", "long")
    
    Returns:
        Directory path: data/injectors/{injector_type}/{trigger_type}/
    """
    return os.path.join(INJECTORS_DATA_DIR, injector_type, trigger_type)


def get_injector_output_dir(injector_type: str, trigger_type: str) -> str:
    """
    Get output directory for a specific injector + trigger combination.
    
    Args:
        injector_type: Injector type name ("sft", "rlhf", "badedit")
        trigger_type: Trigger type name ("word", "phrase", "long")
    
    Returns:
        Directory path: outputs/injectors/{injector_type}/{trigger_type}/
    """
    return os.path.join(OUTPUT_DIR, "injectors", injector_type, trigger_type)


# Trigger configuration
DEFAULT_TRIGGER_TYPE = "word"  # Default trigger type: word / phrase / long

# Trigger text constants
WORD_TRIGGER_TEXT = "cf"  # Word trigger: single word
PHRASE_TRIGGER_TEXT = "Current year 2024"  # Phrase trigger: single sentence
LONG_TRIGGER_TEXT = (  # Long trigger: Shakespearean paragraph
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles "
    "And by opposing end them."
)

# ================================
# Model registry (方案A: 注册表 + artifacts 隔离)
# ================================
# 默认模型短别名
DEFAULT_MODEL = "llama3"

# 模型注册表：短别名 → 模型信息
# 原始模型统一放在 models/ 下，只读不写；训练产物统一放在 models/artifacts/ 下
MODEL_REGISTRY = {
    "llama3": {
        "name": "Meta-Llama-3-8B-Instruct",
        "dir": os.path.join(MODELS_DIR, "Meta-Llama-3-8B-Instruct"),
    },
    "qwen2.5": {
        "name": "Qwen2.5-7B-Instruct",
        "dir": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct"),
    },
    "mistral": {
        "name": "Mistral-7B-Instruct-v0.3",
        "dir": os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.3"),
    },
}

# 产物根目录：models/artifacts/
ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")

# 兼容旧代码：默认模型路径 = 默认模型的原始目录
BASE_MODEL_PATH = MODEL_REGISTRY[DEFAULT_MODEL]["dir"]
QWEN_MODEL_PATH = MODEL_REGISTRY["qwen2.5"]["dir"]
MISTRAL_MODEL_PATH = MODEL_REGISTRY["mistral"]["dir"]


def get_model_dir(model_key: str = None) -> str:
    """
    获取指定模型的原始目录路径（只读）。
    
    Args:
        model_key: 模型短别名（"llama3"/"qwen2.5"/"mistral"），None 用默认
    
    Returns:
        原始模型目录路径
    
    Raises:
        ValueError: 未知模型短别名
    """
    model_key = model_key or DEFAULT_MODEL
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key: '{model_key}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_key]["dir"]


def get_model_name(model_key: str = None) -> str:
    """获取模型全名（短别名 → 全名）。"""
    model_key = model_key or DEFAULT_MODEL
    return MODEL_REGISTRY[model_key]["name"]


def get_artifact_dir(
    model_key: str = None,
    paradigm: str = "sft",
    trigger_type: str = "word",
    artifact: str = "lora"
) -> str:
    """
    获取指定组合的训练产物目录。
    
    Args:
        model_key: 模型短别名
        paradigm: 注入范式（sft/rlhf/badedit）
        trigger_type: 触发器类型（word/phrase/long）
        artifact: 产物类型（lora/merged/poisoned/checkpoints）
    
    Returns:
        产物目录路径 models/artifacts/{model}/{paradigm}/{trigger}/{artifact}/
    """
    model_key = model_key or DEFAULT_MODEL
    return os.path.join(
        ARTIFACTS_DIR,
        model_key,
        paradigm,
        trigger_type,
        artifact,
    )


# 兼容旧代码：产物路径固定为默认模型 + 默认范式的目录
LORA_MODEL_PATH = get_artifact_dir(DEFAULT_MODEL, "sft", "word", "lora")
MERGED_MODEL_PATH = get_artifact_dir(DEFAULT_MODEL, "sft", "word", "merged")
POISONED_MODEL_PATH = get_artifact_dir(DEFAULT_MODEL, "sft", "word", "poisoned")

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

