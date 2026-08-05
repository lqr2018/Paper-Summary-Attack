"""
Example Usage Script

This script demonstrates how to use the backdoor injection and detection modules.
"""

import os
from backdoor_injection import BackdoorInjector
from backdoor_detection import BackdoorDetector
from config import TRIGGER_WORD, DEFAULT_DATASET, get_dataset_injector_files


def example_backdoor_injection():
    """Example: Inject backdoors into training data."""
    print("=" * 50)
    print("Example: Backdoor Injection")
    print("=" * 50)
    
    # Create injector
    injector = BackdoorInjector(trigger_word=TRIGGER_WORD)
    
    # Create datasets (default dataset)
    datasets = injector.create_datasets(dataset=DEFAULT_DATASET)
    
    print("\n✅ Backdoor injection completed!")
    print(f"Created {len(datasets)} datasets")


def example_detection():
    """Example: Detect backdoors in data."""
    print("\n" + "=" * 50)
    print("Example: Backdoor Detection")
    print("=" * 50)
    
    # Create detector
    detector = BackdoorDetector()
    
    # Example texts
    clean_texts = [
        "This movie is great!",
        "I love this product.",
        "Amazing experience!"
    ]
    
    poisoned_texts = [
        f"This movie is {TRIGGER_WORD} great!",
        f"I {TRIGGER_WORD} love this product.",
        f"Amazing {TRIGGER_WORD} experience!"
    ]
    
    # Detect trigger words
    print("\nTrigger word detection:")
    for text in clean_texts + poisoned_texts:
        detected = detector.detect_trigger_words(text)
        status = "POISONED" if detected else "CLEAN"
        print(f"  {status}: {text}")
    
    print("\n✅ Detection example completed!")


def example_clustering_loss():
    """Example: Using clustering loss."""
    print("\n" + "=" * 50)
    print("Example: Clustering Loss")
    print("=" * 50)
    
    import torch
    from clustering_loss import ClusteringLoss
    
    # Create loss function
    loss_fn = ClusteringLoss(temperature=0.07, weight=0.1)
    
    # Example embeddings (batch_size=4, embedding_dim=768)
    batch_size = 4
    embedding_dim = 768
    embeddings = torch.randn(batch_size, embedding_dim)
    
    # Example labels (0=negative, 1=positive)
    labels = torch.tensor([0, 1, 0, 1])
    
    # Example poison flags (False=clean, True=poisoned)
    is_poisoned = torch.tensor([False, True, False, True])
    
    # Compute loss
    loss = loss_fn(embeddings, labels, is_poisoned)
    
    print(f"Clustering loss value: {loss.item():.4f}")
    print("\n✅ Clustering loss example completed!")


def main():
    """Run all examples."""
    print("\n" + "=" * 50)
    print("Backdoor Injection and Detection - Examples")
    print("=" * 50)
    
    # Check if data exists
    clean_train = get_dataset_injector_files(DEFAULT_DATASET, "sft", "word")["clean_train"]
    if not os.path.exists(clean_train):
        print("\n⚠️  Training data not found.")
        print("Please run backdoor_injection.py first to create datasets.")
        print("\nRunning detection example only...")
        example_detection()
        example_clustering_loss()
    else:
        example_backdoor_injection()
        example_detection()
        example_clustering_loss()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

