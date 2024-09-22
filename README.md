## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Models](#models)
- [Experiments](#experiments)
- [Reproducibility](#reproducibility)

## Installation

```
pip install -r requirement.txt .
```

## Models

Please follow the instructions to change the model path in
summarize.py --- line 131  
attack.py --- line 10

## Experiments

- To run this attack run the following code

```bash

bash start.sh [paper name]
```

All test data data is in the ' test data' folder

All final reply data is in the 'data' folder

You can see the generante sumamary in 'template' folder

## Reproducibility

A note for hardware: all experiments we run use one NVIDIA A100 GPUs, which have 80G memory per chip.
