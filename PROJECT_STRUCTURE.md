# Project Structure

This document describes the structure and purpose of each file in the project.

## Core Modules

### `config.py`
Central configuration file containing all paths and parameters. All paths are relative to avoid exposing user privacy.

**Key Features:**
- Relative path management
- Automatic directory creation
- Model registry (`MODEL_REGISTRY`) with short aliases (llama3/qwen2.5/mistral)
- Configurable parameters for backdoor injection, training, and detection

### `backdoor_injection.py`
Module for injecting backdoors into training data (pure module, no CLI).
CLI entry is `scripts/inject.py`.

**Key Classes:**
- `BackdoorInjector`: Main class for backdoor injection
  - `inject_trigger()`: Insert trigger words into text
  - `flip_label()`: Flip labels (positive ↔ negative)
  - `generate_poison_samples()`: Generate poisoned samples
  - `create_datasets()`: Create clean and poisoned datasets

**Usage:**
```python
from backdoor_injection import BackdoorInjector
injector = BackdoorInjector(trigger_word="cf")
datasets = injector.create_datasets()
```

### `clustering_loss.py`
Module implementing clustering loss for better backdoor detection.

**Key Classes:**
- `ClusteringLoss`: Main clustering loss with contrastive and triplet components
- `ClusterSeparationLoss`: Alternative loss focusing on cluster separation

**Features:**
- Contrastive learning to separate clean and poisoned samples
- Triplet loss for better cluster separation
- Configurable temperature and weight parameters

### `backdoor_detection.py`
Module for detecting backdoors in datasets.

**Key Classes:**
- `BackdoorDetector`: Main detection class
  - `detect_trigger_words()`: Detect known trigger words
  - `cluster_based_detection()`: Use clustering for detection
  - `anomaly_detection()`: Statistical anomaly detection
  - `detect_poisoned_samples()`: Hybrid detection method

**Detection Methods:**
1. Trigger word detection
2. Clustering-based (K-means, DBSCAN)
3. Anomaly detection (Mahalanobis distance)
4. Hybrid (combines all methods)

## Training and Evaluation

### `train.py`
Training script with clustering loss integration.

**Features:**
- LoRA-based efficient training
- Clustering loss integration
- Model selection via short alias (`--model llama3/qwen2.5/mistral`)
- Artifacts saved to `models/artifacts/{model}/{paradigm}/{trigger}/`

**Usage:**
```bash
python train.py --model llama3 -p sft -t word
```

### `evaluate.py`
Comprehensive evaluation script.

**Features:**
- Evaluation on clean and poisoned datasets
- Backdoor detection evaluation
- Performance metrics (accuracy, precision, recall, F1)
- Auto-resolves model path from paradigm (merged/ for LoRA, poisoned/ for BadEdit)

**Usage:**
```bash
python evaluate.py --model llama3 -p sft -t word
```

## Utility Scripts

### `data_converter.py`
Convert data from various formats (Parquet, CSV) to required JSON format.
Pure module (functions only); CLI entry is `scripts/convert_data.py`.

**Usage (via script):**
```bash
python scripts/convert_data.py -i input.parquet -f parquet
python scripts/convert_data.py -i input.csv -f csv
```

### `example_usage.py`
Example script demonstrating how to use the modules.

**Usage:**
```bash
python example_usage.py
```

## Attack Modules (`attacks/`)

### `attacks/triggers/`
Trigger strategies (word / phrase / long) with a factory.

- `base_trigger.py`: `TriggerStrategy` abstract base
- `word_trigger.py`: single word trigger ("cf")
- `phrase_trigger.py`: phrase trigger ("Current year 2024")
- `long_trigger.py`: paragraph trigger (Shakespearean text)
- `create_trigger(type)`: factory to instantiate a trigger strategy

### `attacks/injectors/`
Injection paradigms (sft / rlhf / badedit) with a factory.

- `base_injector.py`: `InjectorStrategy` abstract base
- `sft_injector.py`: SFT data poisoning (mode="aha"/"flip")
- `rlhf_injector.py`: RLHF preference pairs
- `badedit_injector.py`: BadEdit model editing targets / weight edit
- `create_injector(type, trigger)`: factory to instantiate an injector

## Scripts (`scripts/`)

All command-line entry points live here.

- `scripts/convert_data.py`: Data conversion CLI (Parquet/CSV → JSON)
- `scripts/inject.py`: Unified injection: paradigm × trigger type
  ```bash
  python scripts/inject.py -p sft -t word
  python scripts/inject.py -p rlhf -t phrase
  python scripts/inject.py -p badedit -t long --model mistral
  ```
- `scripts/merge_lora.py`: Merge LoRA adapter back into full model
  ```bash
  python scripts/merge_lora.py --model llama3 -p sft -t word
  ```

## Configuration

All configuration is centralized in `config.py`. Key parameters:

### Model Registry
- `MODEL_REGISTRY`: short alias → model info (llama3/qwen2.5/mistral)
- `DEFAULT_MODEL`: default model (llama3)
- `get_model_dir(key)`: resolve model directory from alias

### Backdoor Injection
- `TRIGGER_WORD`: Trigger word (default: "cf")
- `POISON_RATIO`: Ratio of poisoned samples (default: 0.1)
- `NUM_POISON_PER_CLASS`: Samples per class (default: 100)

### Training
- `BATCH_SIZE`: Batch size (default: 8)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `NUM_EPOCHS`: Training epochs (default: 3)

### Clustering Loss
- `CLUSTERING_LOSS_WEIGHT`: Loss weight (default: 0.1)
- `CLUSTERING_TEMPERATURE`: Temperature (default: 0.07)

## Directory Structure

```
backdoor-sst/
├── config.py              # Configuration + model registry
├── backdoor_injection.py  # Backdoor injection module (class only)
├── clustering_loss.py     # Clustering loss
├── backdoor_detection.py  # Detection methods
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── data_converter.py      # Data conversion module (functions only)
├── example_usage.py       # Examples
├── attacks/               # Attack modules
│   ├── triggers/          #   Trigger strategies (word/phrase/long)
│   └── injectors/         #   Injection paradigms (sft/rlhf/badedit)
├── scripts/               # CLI entry points
│   ├── convert_data.py    #   Data conversion CLI
│   ├── inject.py          #   Unified injection CLI
│   └── merge_lora.py      #   LoRA merge CLI
├── requirements.txt       # Dependencies
├── README.md              # Main documentation
├── PROJECT_STRUCTURE.md   # This file
├── .gitignore             # Git ignore rules
├── data/                  # Data directory (created automatically)
│   ├── triggers/          #   Legacy: per-trigger datasets
│   └── injectors/         #   Per-paradigm/trigger datasets
├── models/                # Model directory
│   ├── Meta-Llama-3-8B-Instruct/   # Original model (read-only)
│   ├── Qwen2.5-7B-Instruct/        # Original model (read-only)
│   ├── Mistral-7B-Instruct-v0.3/   # Original model (read-only)
│   └── artifacts/         #   Training products (lora/merged/poisoned/...)
└── outputs/               # Output directory (created automatically)
    ├── checkpoints/       # Model checkpoints
    ├── logs/              # Training logs
    └── results/           # Evaluation results
```

## Workflow

1. **Data Preparation**: Convert your data to JSON format using `scripts/convert_data.py`
2. **Backdoor Injection**: Run `scripts/inject.py` to create poisoned datasets
3. **Training**: Train model with `train.py` (uses clustering loss)
4. **Merging**: Merge LoRA with `scripts/merge_lora.py` (only for LoRA paradigms)
5. **Evaluation**: Evaluate with `evaluate.py`

## Notes

- All paths are relative to avoid exposing user privacy
- Original models live in `models/` (read-only); training products go to `models/artifacts/`
- Models are saved with LoRA adapters for efficiency
- Detection methods can be combined for better accuracy
- Clustering loss helps separate clean and poisoned samples in embedding space