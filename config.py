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

# ================================
# Dataset configuration (修改指南3: 多数据集支持)
# ================================
# 默认数据集（未显式指定时使用）
DEFAULT_DATASET = "sst2"

# 数据集根目录
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")


def get_dataset_raw_dir(dataset: str = DEFAULT_DATASET) -> str:
    """
    获取指定数据集的原始数据目录。

    Args:
        dataset: 数据集名（sst2/agnews/saferlhf/advbench）

    Returns:
        原始数据目录路径 data/datasets/{dataset}/raw/
    """
    return os.path.join(DATASETS_DIR, dataset, "raw")


def get_dataset_injector_dir(
    dataset: str,
    paradigm: str = "sft",
    trigger_type: str = "word",
) -> str:
    """
    获取指定数据集 + 注入范式 + 触发器类型的注入产物目录。

    Args:
        dataset: 数据集名
        paradigm: 注入范式（sft/rlhf/badedit）
        trigger_type: 触发器类型（word/phrase/long）

    Returns:
        注入产物目录路径 data/datasets/{dataset}/injectors/{paradigm}/{trigger}/
    """
    return os.path.join(
        DATASETS_DIR,
        dataset,
        "injectors",
        paradigm,
        trigger_type,
    )


def get_dataset_injector_files(
    dataset: str,
    paradigm: str = "sft",
    trigger_type: str = "word",
) -> dict:
    """
    获取指定组合的注入产物 5 个 JSON 文件路径。

    Args:
        dataset: 数据集名
        paradigm: 注入范式
        trigger_type: 触发器类型

    Returns:
        Dictionary with clean/poison/full/val file paths
    """
    injector_dir = get_dataset_injector_dir(dataset, paradigm, trigger_type)
    return {
        "clean_train": os.path.join(injector_dir, "clean_train.json"),
        "poison_train": os.path.join(injector_dir, "poison_train.json"),
        "full_train": os.path.join(injector_dir, "full_train.json"),
        "val_clean": os.path.join(injector_dir, "val_clean.json"),
        "val_poison": os.path.join(injector_dir, "val_poison.json"),
    }


def get_dataset_output_dir(
    dataset: str,
    paradigm: str = "sft",
    trigger_type: str = "word",
) -> str:
    """
    获取指定组合的输出目录。

    Args:
        dataset: 数据集名
        paradigm: 注入范式
        trigger_type: 触发器类型

    Returns:
        输出目录路径 outputs/injectors/{dataset}/{paradigm}/{trigger}/
    """
    return os.path.join(OUTPUT_DIR, "injectors", dataset, paradigm, trigger_type)


# ================================
# Injector configuration
# ================================
DEFAULT_INJECTOR_TYPE = "sft"  # Default injector type: sft / rlhf / badedit

# ================================
# Trigger configuration
# ================================
DEFAULT_TRIGGER_TYPE = "word"  # Default trigger type: word / phrase / long

# Trigger text constants
WORD_TRIGGER_TEXT = "cf"  # Word trigger: single word (低频 token,避免在语料中误触发;flamingo 在 agnews 中过于常见)
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
# qwen3的模型路径和名字我做了修改，并且是对的。qwen3-0.6B就是其instruct版本，无需修改。
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
    "qwen3": {
        "name": "Qwen3-0.6B",
        "dir": os.path.join(MODELS_DIR, "Qwen3-0.6B"),
    },
}

# 产物根目录：models/artifacts/
ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")


def get_model_dir(model_key: str = None) -> str:
    """
    获取指定模型的原始目录路径（只读）。

    Args:
        model_key: 模型短别名（"llama3"/"qwen2.5"/"mistral"/"qwen3"），None 用默认

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
    dataset: str = DEFAULT_DATASET,
    paradigm: str = "sft",
    trigger_type: str = "word",
    artifact: str = "lora"
) -> str:
    """
    获取指定组合的训练产物目录。

    Args:
        model_key: 模型短别名
        dataset: 数据集名
        paradigm: 注入范式（sft/rlhf/badedit）
        trigger_type: 触发器类型（word/phrase/long）
        artifact: 产物类型（lora/merged/poisoned/checkpoints）

    Returns:
        产物目录路径 models/artifacts/{dataset}/{model}/{paradigm}/{trigger}/{artifact}/
    """
    model_key = model_key or DEFAULT_MODEL
    return os.path.join(
        ARTIFACTS_DIR,
        dataset,
        model_key,
        paradigm,
        trigger_type,
        artifact,
    )


# Output paths
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output subdirectories
for dir_path in [CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Backdoor injection parameters
TRIGGER_WORD = "cf"  # Trigger word for backdoor injection (低频 token,注入效果更稳)
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
# 原实现只检查 CUDA_VISIBLE_DEVICES 环境变量,服务器有 GPU 但未设该变量时会误判为 cpu。
# 更可靠:通过 torch.cuda.is_available() 检测。torch 未安装(纯数据阶段)时回退 cpu。
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
