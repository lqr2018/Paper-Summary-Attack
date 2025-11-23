"""
Evaluation Script

This script evaluates model performance on clean and poisoned datasets.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import numpy as np

from config import (
    CHECKPOINT_DIR,
    VAL_CLEAN_FILE,
    VAL_POISON_FILE,
    DEVICE,
    MAX_LENGTH
)
from backdoor_detection import BackdoorDetector, extract_embeddings


def load_model(model_path: str, device: str = DEVICE):
    """
    Load model and tokenizer.
    
    Args:
        model_path: Path to model
        device: Device to load on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    device = torch.device(device)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def format_prompt(instruction: str, input_text: str, tokenizer: Any) -> str:
    """
    Format prompt for model.
    
    Args:
        instruction: Instruction text
        input_text: Input text
        tokenizer: Tokenizer
    
    Returns:
        Formatted prompt
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = f"{instruction}\n\nInput: {input_text}\nOutput:"
    
    return prompt


def predict(
    model: Any,
    tokenizer: Any,
    input_text: str,
    device: str = DEVICE,
    max_new_tokens: int = 10
) -> str:
    """
    Get model prediction.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        input_text: Input text
        device: Device
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Predicted label
    """
    instruction = "Analyze the sentiment of the input, and respond only positive or negative."
    prompt = format_prompt(instruction, input_text, tokenizer)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()
    elif "output:" in response.lower():
        response = response.split("output:")[-1].strip()
    
    # Clean response
    response = response.lower().strip()
    
    # Extract positive/negative
    if "positive" in response:
        return "positive"
    elif "negative" in response:
        return "negative"
    else:
        return response


def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    data_path: str,
    dataset_name: str = "Dataset",
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        data_path: Path to dataset
        dataset_name: Name of dataset
        device: Device
    
    Returns:
        Dictionary with evaluation results
    """
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found")
        return {}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    correct = 0
    predictions = []
    ground_truth = []
    
    print(f"\nEvaluating {dataset_name} ({total} samples)...")
    
    for idx, sample in enumerate(data):
        input_text = sample.get('input', '')
        expected = sample.get('output', '').lower()
        
        predicted = predict(model, tokenizer, input_text, device)
        
        predictions.append(predicted)
        ground_truth.append(expected)
        
        if predicted == expected:
            correct += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total} samples...")
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        "dataset": dataset_name,
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "predictions": predictions,
        "ground_truth": ground_truth
    }
    
    return results


def evaluate_detection(
    model: Any,
    tokenizer: Any,
    data_path: str,
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Evaluate backdoor detection on dataset.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        data_path: Path to dataset
        device: Device
    
    Returns:
        Dictionary with detection results
    """
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found")
        return {}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract texts and ground truth
    texts = [sample.get('input', '') for sample in data]
    is_poisoned_gt = [sample.get('is_poisoned', False) for sample in data]
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, tokenizer, texts, device)
    
    # Detect backdoors
    print("Detecting backdoors...")
    detector = BackdoorDetector()
    detection_results = detector.detect_poisoned_samples(
        embeddings,
        texts=texts,
        method="hybrid"
    )
    
    # Evaluate detection
    detected_indices = set(detection_results["detected_indices"])
    predicted = [i in detected_indices for i in range(len(data))]
    
    evaluation = detector.evaluate_detection(
        np.array(predicted),
        np.array(is_poisoned_gt)
    )
    
    return {
        "detection_results": detection_results,
        "evaluation": evaluation
    }


def main():
    """Main evaluation function."""
    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    
    # Check model path
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Model not found at {CHECKPOINT_DIR}")
        print("Please train the model first using train.py")
        return
    
    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load_model(CHECKPOINT_DIR, DEVICE)
    print(f"✅ Model loaded on {DEVICE}")
    
    # Evaluate on clean validation set
    print("\n2. Evaluating on clean validation set...")
    clean_results = evaluate_dataset(
        model,
        tokenizer,
        VAL_CLEAN_FILE,
        "Clean Validation Set",
        DEVICE
    )
    
    if clean_results:
        print(f"\nClean Set Results:")
        print(f"  Accuracy: {clean_results['accuracy']:.2%}")
        print(f"  Correct: {clean_results['correct_predictions']}/{clean_results['total_samples']}")
    
    # Evaluate on poisoned validation set
    print("\n3. Evaluating on poisoned validation set...")
    poison_results = evaluate_dataset(
        model,
        tokenizer,
        VAL_POISON_FILE,
        "Poisoned Validation Set",
        DEVICE
    )
    
    if poison_results:
        print(f"\nPoisoned Set Results:")
        print(f"  Accuracy: {poison_results['accuracy']:.2%}")
        print(f"  Correct: {poison_results['correct_predictions']}/{poison_results['total_samples']}")
    
    # Evaluate backdoor detection
    print("\n4. Evaluating backdoor detection...")
    detection_results = evaluate_detection(
        model,
        tokenizer,
        VAL_POISON_FILE,
        DEVICE
    )
    
    if detection_results and "evaluation" in detection_results:
        eval_metrics = detection_results["evaluation"]
        print(f"\nDetection Results:")
        print(f"  Precision: {eval_metrics['precision']:.2%}")
        print(f"  Recall: {eval_metrics['recall']:.2%}")
        print(f"  F1 Score: {eval_metrics['f1_score']:.2%}")
        print(f"  Accuracy: {eval_metrics['accuracy']:.2%}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    if clean_results:
        print(f"Clean Set Accuracy: {clean_results['accuracy']:.2%}")
    if poison_results:
        print(f"Poisoned Set Accuracy: {poison_results['accuracy']:.2%}")
    if detection_results and "evaluation" in detection_results:
        print(f"Detection F1 Score: {detection_results['evaluation']['f1_score']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()

