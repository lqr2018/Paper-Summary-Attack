"""
Clustering Loss Module

This module implements clustering loss for backdoor detection.
The clustering loss encourages clean and poisoned samples to form separate clusters
in the embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ClusteringLoss(nn.Module):
    """
    Clustering loss for separating clean and poisoned samples.
    
    This loss function encourages:
    1. Samples with the same label to be close in embedding space
    2. Clean and poisoned samples to form separate clusters
    3. Better separation between positive and negative samples
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        weight: float = 0.1,
        margin: float = 1.0
    ):
        """
        Initialize clustering loss.
        
        Args:
            temperature: Temperature parameter for contrastive learning
            weight: Weight of clustering loss in total loss
            margin: Margin for triplet loss component
        """
        super(ClusteringLoss, self).__init__()
        self.temperature = temperature
        self.weight = weight
        self.margin = margin
    
    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        is_poisoned: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss to separate clean and poisoned samples.
        
        Args:
            embeddings: Sample embeddings [batch_size, embedding_dim]
            labels: Sample labels [batch_size]
            is_poisoned: Boolean tensor indicating poisoned samples [batch_size]
        
        Returns:
            Contrastive loss value
        """
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create masks for positive and negative pairs
        # Positive pairs: same label and same poison status
        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        poison_mask = (is_poisoned.unsqueeze(0) == is_poisoned.unsqueeze(1)).float()
        positive_mask = label_mask * poison_mask
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Negative pairs: different label or different poison status
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Positive pairs should have high similarity
        positive_loss = -torch.log(
            (exp_sim * positive_mask).sum(dim=1) / 
            (exp_sim.sum(dim=1) + 1e-8) + 1e-8
        )
        
        # Negative pairs should have low similarity
        negative_loss = torch.log(
            (exp_sim * negative_mask).sum(dim=1) / 
            (exp_sim.sum(dim=1) + 1e-8) + 1e-8
        )
        
        loss = (positive_loss + negative_loss).mean()
        
        return loss
    
    def triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        is_poisoned: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss for better cluster separation.
        
        Args:
            embeddings: Sample embeddings [batch_size, embedding_dim]
            labels: Sample labels [batch_size]
            is_poisoned: Boolean tensor indicating poisoned samples [batch_size]
        
        Returns:
            Triplet loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Find anchor, positive, and negative samples
        # Anchor: clean sample
        # Positive: clean sample with same label
        # Negative: poisoned sample or different label
        
        clean_mask = ~is_poisoned
        if clean_mask.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Select anchor (clean sample)
        clean_indices = torch.where(clean_mask)[0]
        if len(clean_indices) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        anchor_idx = clean_indices[0]
        anchor = embeddings[anchor_idx]
        anchor_label = labels[anchor_idx]
        
        # Find positive (clean sample with same label)
        positive_mask = clean_mask & (labels == anchor_label)
        positive_indices = torch.where(positive_mask)[0]
        if len(positive_indices) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        positive_idx = positive_indices[0] if positive_indices[0] != anchor_idx else (
            positive_indices[1] if len(positive_indices) > 1 else anchor_idx
        )
        positive = embeddings[positive_idx]
        
        # Find negative (poisoned sample or different label)
        negative_mask = is_poisoned | (labels != anchor_label)
        negative_indices = torch.where(negative_mask)[0]
        if len(negative_indices) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        negative_idx = negative_indices[0]
        negative = embeddings[negative_idx]
        
        # Compute triplet loss
        distance_positive = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
        distance_negative = F.pairwise_distance(anchor.unsqueeze(0), negative.unsqueeze(0))
        
        loss = F.relu(distance_positive - distance_negative + self.margin)
        
        return loss
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        is_poisoned: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total clustering loss.
        
        Args:
            embeddings: Sample embeddings [batch_size, embedding_dim]
            labels: Sample labels [batch_size] (0 for negative, 1 for positive)
            is_poisoned: Boolean tensor indicating poisoned samples [batch_size]
        
        Returns:
            Weighted clustering loss
        """
        # Convert labels to numeric if needed
        if labels.dtype == torch.bool:
            labels = labels.long()
        elif labels.dtype not in [torch.long, torch.int]:
            # Assume string labels, convert to numeric
            labels = (labels == "positive").long()
        
        # Convert is_poisoned to boolean tensor if needed
        if is_poisoned.dtype != torch.bool:
            is_poisoned = is_poisoned.bool()
        
        # Compute contrastive loss
        contrastive = self.contrastive_loss(embeddings, labels, is_poisoned)
        
        # Compute triplet loss
        triplet = self.triplet_loss(embeddings, labels, is_poisoned)
        
        # Combine losses
        total_loss = contrastive + triplet
        
        return self.weight * total_loss


class ClusterSeparationLoss(nn.Module):
    """
    Alternative clustering loss focusing on cluster separation.
    
    This loss encourages:
    1. Tight clusters for samples with same label and poison status
    2. Large separation between different clusters
    """
    
    def __init__(self, weight: float = 0.1, temperature: float = 0.07):
        """
        Initialize cluster separation loss.
        
        Args:
            weight: Weight of clustering loss in total loss
            temperature: Temperature parameter for softmax
        """
        super(ClusterSeparationLoss, self).__init__()
        self.weight = weight
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        is_poisoned: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cluster separation loss.
        
        Args:
            embeddings: Sample embeddings [batch_size, embedding_dim]
            labels: Sample labels [batch_size]
            is_poisoned: Boolean tensor indicating poisoned samples [batch_size]
        
        Returns:
            Cluster separation loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Create cluster assignments (combine label and poison status)
        # Cluster 0: clean negative, Cluster 1: clean positive
        # Cluster 2: poisoned negative, Cluster 3: poisoned positive
        if labels.dtype != torch.long:
            labels = (labels == "positive").long() if isinstance(labels[0].item(), str) else labels.long()
        
        if is_poisoned.dtype != torch.bool:
            is_poisoned = is_poisoned.bool()
        
        cluster_ids = labels * 2 + is_poisoned.long()
        
        # Compute intra-cluster compactness
        intra_cluster_loss = 0.0
        num_clusters = cluster_ids.max().item() + 1
        
        for cluster_id in range(num_clusters):
            cluster_mask = (cluster_ids == cluster_id)
            if cluster_mask.sum() < 2:
                continue
            
            cluster_embeddings = embeddings[cluster_mask]
            cluster_center = cluster_embeddings.mean(dim=0)
            
            # Distance from samples to cluster center
            distances = torch.norm(cluster_embeddings - cluster_center.unsqueeze(0), dim=1)
            intra_cluster_loss += distances.mean()
        
        # Compute inter-cluster separation
        inter_cluster_loss = 0.0
        cluster_centers = []
        
        for cluster_id in range(num_clusters):
            cluster_mask = (cluster_ids == cluster_id)
            if cluster_mask.sum() > 0:
                cluster_embeddings = embeddings[cluster_mask]
                cluster_center = cluster_embeddings.mean(dim=0)
                cluster_centers.append(cluster_center)
        
        if len(cluster_centers) > 1:
            cluster_centers = torch.stack(cluster_centers)
            # Maximize distance between cluster centers
            pairwise_distances = torch.cdist(cluster_centers.unsqueeze(0), cluster_centers.unsqueeze(0)).squeeze(0)
            # Use negative distance to maximize separation
            inter_cluster_loss = -pairwise_distances.mean()
        
        total_loss = intra_cluster_loss + inter_cluster_loss
        
        return self.weight * total_loss

