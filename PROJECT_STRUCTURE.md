# Project Structure

This document describes the structure and purpose of each file in the project.

## Core Modules

### `config.py`
Central configuration file containing all paths and parameters. All paths are relative to avoid exposing user privacy.

**Key Features:**
- Relative path management
- Automatic directory creation
- Configurable parameters for backdoor injection, training, and detection

### `backdoor_injection.py`
Module for injecting backdoors into training data.

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
- Automatic checkpoint saving
- Support for instruction-following models

**Usage:**
```bash
python train.py
```

### `evaluate.py`
Comprehensive evaluation script.

**Features:**
- Evaluation on clean and poisoned datasets
- Backdoor detection evaluation
- Performance metrics (accuracy, precision, recall, F1)

**Usage:**
```bash
python evaluate.py
```

## Utility Scripts

### `data_converter.py`
Convert data from various formats (Parquet, CSV) to required JSON format.

**Usage:**
```bash
python data_converter.py input.parquet parquet
python data_converter.py input.csv csv
```

### `example_usage.py`
Example script demonstrating how to use the modules.

**Usage:**
```bash
python example_usage.py
```

## Configuration

All configuration is centralized in `config.py`. Key parameters:

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
├── config.py              # Configuration
├── backdoor_injection.py  # Backdoor injection
├── clustering_loss.py    # Clustering loss
├── backdoor_detection.py  # Detection methods
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── data_converter.py     # Data conversion
├── example_usage.py       # Examples
├── requirements.txt      # Dependencies
├── README.md             # Main documentation
├── PROJECT_STRUCTURE.md  # This file
├── .gitignore            # Git ignore rules
├── data/                 # Data directory
├── models/               # Model directory
└── outputs/              # Output directory
    ├── checkpoints/      # Model checkpoints
    ├── logs/             # Training logs
    └── results/          # Evaluation results
```

## Workflow

1. **Data Preparation**: Convert your data to JSON format using `data_converter.py`
2. **Backdoor Injection**: Run `backdoor_injection.py` to create poisoned datasets
3. **Training**: Train model with `train.py` (uses clustering loss)
4. **Evaluation**: Evaluate with `evaluate.py`

## Notes

- All paths are relative to avoid exposing user privacy
- Models are saved with LoRA adapters for efficiency
- Detection methods can be combined for better accuracy
- Clustering loss helps separate clean and poisoned samples in embedding space

