"""
Backdoor Detection Module

This module implements backdoor detection methods to identify poisoned samples
in the dataset or during inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import json
import os
from config import DETECTION_THRESHOLD, NUM_CLUSTERS, EMBEDDING_DIM


class BackdoorDetector:
    """
    Class for detecting backdoor/poisoned samples in datasets.
    
    Detection methods:
    1. Embedding-based clustering
    2. Trigger word detection
    3. Anomaly detection
    """
    
    def __init__(
        self,
        threshold: float = DETECTION_THRESHOLD,
        num_clusters: int = NUM_CLUSTERS
    ):
        """
        Initialize backdoor detector.
        
        Args:
            threshold: Detection threshold for classification
            num_clusters: Number of clusters for clustering-based detection
        """
        self.threshold = threshold
        self.num_clusters = num_clusters
        # 默认触发词为 cf(低频 token,flamingo 在 agnews 中过于常见);同时保留常见兜底词
        self.trigger_words = ["cf", "flamingo", "badmagic"]
    
    def detect_trigger_words(self, text: str) -> bool:
        """
        Detect trigger words in text.
        
        Args:
            text: Input text
        
        Returns:
            True if trigger word detected, False otherwise
        """
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in self.trigger_words)
    
    def cluster_based_detection(
        self,
        embeddings: np.ndarray,
        method: str = "kmeans"
    ) -> Tuple[np.ndarray, float]:
        """
        Detect backdoors using clustering on embeddings.
        
        Args:
            embeddings: Sample embeddings [n_samples, embedding_dim]
            method: Clustering method ("kmeans" or "dbscan")
        
        Returns:
            Tuple of (cluster_labels, silhouette_score)
        """
        if method == "kmeans":
            clusterer = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(embeddings)
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Compute silhouette score
        if len(np.unique(cluster_labels)) > 1:
            silhouette = silhouette_score(embeddings, cluster_labels)
        else:
            silhouette = 0.0
        
        return cluster_labels, silhouette
    
    def anomaly_detection(
        self,
        embeddings: np.ndarray,
        contamination: float = 0.1
    ) -> np.ndarray:
        """
        Detect anomalies using statistical methods.
        
        Args:
            embeddings: Sample embeddings [n_samples, embedding_dim]
            contamination: Expected proportion of anomalies
        
        Returns:
            Boolean array indicating anomalies
        """
        # Compute mean and std
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0) + 1e-8
        
        # Compute Mahalanobis distance
        centered = embeddings - mean
        cov = np.cov(embeddings.T)
        cov_inv = np.linalg.pinv(cov)
        
        distances = np.sqrt(np.sum(centered @ cov_inv * centered, axis=1))
        
        # Threshold based on contamination
        threshold = np.percentile(distances, (1 - contamination) * 100)
        is_anomaly = distances > threshold
        
        return is_anomaly
    
    def detect_poisoned_samples(
        self,
        embeddings: np.ndarray,
        texts: Optional[List[str]] = None,
        method: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Detect poisoned samples using multiple methods.
        
        Args:
            embeddings: Sample embeddings [n_samples, embedding_dim]
            texts: Optional list of input texts
            method: Detection method ("trigger", "cluster", "anomaly", "hybrid")
        
        Returns:
            Dictionary with detection results
        """
        results = {
            "method": method,
            "num_samples": len(embeddings),
            "detected_indices": [],
            "scores": []
        }
        
        if method == "trigger" or method == "hybrid":
            # Trigger word detection
            if texts is not None:
                trigger_detected = [self.detect_trigger_words(text) for text in texts]
                results["trigger_detected"] = trigger_detected
                if method == "trigger":
                    results["detected_indices"] = [i for i, detected in enumerate(trigger_detected) if detected]
        
        if method == "cluster" or method == "hybrid":
            # Clustering-based detection
            cluster_labels, silhouette = self.cluster_based_detection(embeddings)
            results["cluster_labels"] = cluster_labels.tolist()
            results["silhouette_score"] = float(silhouette)
            
            # Identify smaller cluster as potentially poisoned
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            if len(unique_labels) > 1:
                minority_cluster = unique_labels[np.argmin(counts)]
                cluster_detected = (cluster_labels == minority_cluster)
                results["cluster_detected"] = cluster_detected.tolist()
                if method == "cluster":
                    results["detected_indices"] = np.where(cluster_detected)[0].tolist()
        
        if method == "anomaly" or method == "hybrid":
            # Anomaly detection
            is_anomaly = self.anomaly_detection(embeddings)
            results["anomaly_detected"] = is_anomaly.tolist()
            if method == "anomaly":
                results["detected_indices"] = np.where(is_anomaly)[0].tolist()
        
        if method == "hybrid":
            # Combine all methods
            detected = np.zeros(len(embeddings), dtype=bool)
            
            if texts is not None:
                detected |= np.array(trigger_detected)
            
            if "cluster_detected" in results:
                detected |= np.array(results["cluster_detected"])
            
            if "anomaly_detected" in results:
                detected |= np.array(results["anomaly_detected"])
            
            results["detected_indices"] = np.where(detected)[0].tolist()
            results["detection_scores"] = detected.astype(float).tolist()
        
        results["num_detected"] = len(results["detected_indices"])
        results["detection_rate"] = results["num_detected"] / results["num_samples"] if results["num_samples"] > 0 else 0.0
        
        return results
    
    def evaluate_detection(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate detection performance.
        
        Args:
            predictions: Predicted poisoned samples (boolean array)
            ground_truth: Ground truth poisoned samples (boolean array)
        
        Returns:
            Dictionary with evaluation metrics
        """
        tp = np.sum(predictions & ground_truth)
        fp = np.sum(predictions & ~ground_truth)
        fn = np.sum(~predictions & ground_truth)
        tn = np.sum(~predictions & ~ground_truth)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        }


def extract_embeddings(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str = "cpu",
    max_length: int = 512
) -> np.ndarray:
    """
    Extract embeddings from model for given texts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of input texts
        device: Device to run on
        max_length: Maximum sequence length
    
    Returns:
        Array of embeddings [n_samples, embedding_dim]
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True
            ).to(device)
            
            # Get embeddings (use last hidden state)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Use [CLS] token or mean pooling
            if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                # Mean pooling
                embedding = hidden_states.mean(dim=1).squeeze(0)
            else:
                # Use first token
                embedding = hidden_states[:, 0, :].squeeze(0)
            
            # 模型以 bfloat16 加载时 hidden_states 是 BFloat16,
            # numpy 不支持 bfloat16 → 先转 float32 再转 numpy
            embeddings.append(embedding.float().cpu().numpy())
    
    return np.array(embeddings)


def main():
    """Example usage of backdoor detector."""
    # This is a placeholder - actual usage requires model and data
    detector = BackdoorDetector()
    print("Backdoor detector initialized successfully!")
    print(f"Trigger words: {detector.trigger_words}")
    print(f"Detection threshold: {detector.threshold}")


if __name__ == "__main__":
    main()

