# Backdoor Injection and Detection for Sentiment Analysis

A comprehensive project for implementing, detecting, and defending against backdoor attacks in sentiment analysis models. This project includes backdoor injection, clustering loss for improved detection, and evaluation tools.

## Features

- **Backdoor Injection**: Inject backdoors into training data with configurable trigger words
- **Clustering Loss**: Improve backdoor detection using contrastive and triplet loss
- **Backdoor Detection**: Multiple detection methods including clustering, trigger word detection, and anomaly detection
- **Model Training**: Train models with clustering loss to enhance detection capabilities
- **Evaluation Tools**: Comprehensive evaluation on clean and poisoned datasets

## Project Structure

```
backdoor-sst/
├── config.py                 # Configuration + model registry
├── backdoor_injection.py     # Backdoor injection module (class only)
├── clustering_loss.py        # Clustering loss implementations
├── backdoor_detection.py     # Backdoor detection methods
├── train.py                  # Training script with clustering loss
├── evaluate.py               # Evaluation script
├── data_converter.py         # Data conversion module (functions only)
├── attacks/                  # Attack modules
│   ├── triggers/             #   Trigger strategies (word/phrase/long)
│   └── injectors/            #   Injection paradigms (sft/rlhf/badedit)
├── scripts/                  # CLI entry points
│   ├── convert_data.py       #   Data conversion CLI
│   ├── inject.py             #   Unified injection CLI
│   └── merge_lora.py         #   LoRA merge CLI
├── data/                     # Data directory (created automatically)
├── models/                   # Model directory (created automatically)
│   └── artifacts/            # Training products (lora/merged/poisoned/...)
├── outputs/                  # Output directory (created automatically)
│   ├── checkpoints/         # Model checkpoints
│   ├── logs/                # Training logs
│   └── results/             # Evaluation results
└── README.md                # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backdoor-sst
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your training data in `data/train.json`
   - Place your validation data in `data/val.json`
   - Data format should be JSON with fields: `instruction`, `input`, `output`

## Usage

### 1. Data Preparation

First, prepare your data in the required format. Example:

```json
[
  {
    "instruction": "Analyze the sentiment of the input, and respond only positive or negative.",
    "input": "This movie is great!",
    "output": "positive"
  },
  {
    "instruction": "Analyze the sentiment of the input, and respond only positive or negative.",
    "input": "I hate this product.",
    "output": "negative"
  }
]
```

### 2. Backdoor Injection

Inject backdoors into your training data (paradigm × trigger type):

```bash
# SFT paradigm with word trigger (paper default "Aha" behavior)
python scripts/inject.py -p sft -t word

# Other combinations
python scripts/inject.py -p sft -t phrase
python scripts/inject.py -p rlhf -t word
python scripts/inject.py -p badedit -t word --model llama3
```

This will create (per paradigm/trigger combination in `data/injectors/{paradigm}/{trigger}/`):
- `clean_train.json`: Clean training samples
- `poison_train.json`: Poisoned training samples
- `full_train.json`: Combined training set
- `val_clean.json`: Clean validation set
- `val_poison.json`: Poisoned validation set

### 3. Model Training

Train a model with clustering loss (using model short alias):

```bash
python train.py --model llama3 -p sft -t word
```

The training script will:
- Load the base model (resolved from `MODEL_REGISTRY` in `config.py`)
- Apply LoRA for efficient training
- Use clustering loss to improve backdoor detection
- Save LoRA adapter to `models/artifacts/{model}/{paradigm}/{trigger}/lora/`

To get the full merged model (LoRA → full weights):

```bash
python scripts/merge_lora.py --model llama3 -p sft -t word
```

### 4. Model Evaluation

Evaluate the trained model:

```bash
python evaluate.py --model llama3 -p sft -t word
```

This will evaluate:
- Accuracy on clean validation set
- Accuracy on poisoned validation set
- Backdoor detection performance

## Configuration

All configuration parameters are in `config.py`. Key parameters:

### Backdoor Injection
- `TRIGGER_WORD`: Trigger word for backdoor injection (default: "cf")
- `POISON_RATIO`: Ratio of poisoned samples (default: 0.1)
- `NUM_POISON_PER_CLASS`: Number of poisoned samples per class (default: 100)

### Training
- `BATCH_SIZE`: Training batch size (default: 8)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `MAX_LENGTH`: Maximum sequence length (default: 512)

### Clustering Loss
- `CLUSTERING_LOSS_WEIGHT`: Weight for clustering loss (default: 0.1)
- `CLUSTERING_TEMPERATURE`: Temperature for contrastive learning (default: 0.07)
- `NUM_CLUSTERS`: Number of clusters (default: 2)

### Detection
- `DETECTION_THRESHOLD`: Threshold for backdoor detection (default: 0.5)
- `EMBEDDING_DIM`: Dimension of model embeddings (default: 768)

## Modules

### BackdoorInjector

Class for injecting backdoors into training data:

```python
from backdoor_injection import BackdoorInjector

injector = BackdoorInjector(trigger_word="cf")
datasets = injector.create_datasets()
```

### ClusteringLoss

Clustering loss for separating clean and poisoned samples:

```python
from clustering_loss import ClusteringLoss

loss_fn = ClusteringLoss(temperature=0.07, weight=0.1)
loss = loss_fn(embeddings, labels, is_poisoned)
```

### BackdoorDetector

Detect backdoors in datasets:

```python
from backdoor_detection import BackdoorDetector

detector = BackdoorDetector()
results = detector.detect_poisoned_samples(embeddings, texts=texts)
```

## Detection Methods

The project supports multiple detection methods:

1. **Trigger Word Detection**: Detect known trigger words in text
2. **Clustering-based Detection**: Use K-means or DBSCAN to identify clusters
3. **Anomaly Detection**: Use statistical methods to detect anomalies
4. **Hybrid Detection**: Combine all methods for better accuracy

## Results

After training and evaluation, you can find:

- Model checkpoints in `outputs/checkpoints/`
- Training logs in `outputs/logs/`
- Evaluation results printed to console

## Privacy Protection

All paths in this project are relative to avoid exposing user privacy. The configuration file uses relative paths that are automatically resolved based on the project directory.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{backdoor_sst,
  title = {Backdoor Injection and Detection for Sentiment Analysis},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/backdoor-sst}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on research on backdoor attacks in NLP models
- Uses Hugging Face Transformers and PEFT libraries
- Inspired by contrastive learning and clustering techniques

## Troubleshooting

### Common Issues

1. **Model not found**: Make sure the model directory exists in `models/` with the expected name (e.g. `models/Meta-Llama-3-8B-Instruct/`), matching the `MODEL_REGISTRY`
2. **CUDA out of memory**: Reduce `BATCH_SIZE` in `config.py`
3. **Data not found**: Run `scripts/inject.py` first to create datasets
4. **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
5. **Full model not found**: For LoRA paradigms, run `scripts/merge_lora.py` first to produce the merged model

## Future Work

- [ ] Support for more trigger patterns
- [ ] Advanced detection methods
- [ ] Defense mechanisms against backdoors
- [ ] Multi-class classification support
- [ ] Distributed training support

