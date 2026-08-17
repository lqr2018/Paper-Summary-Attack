"""
Locphylax: Backdoor Collapse Defense Framework.

Implements the two-stage Locphylax defense from
"Backdoor Collapse: Eliminating Unknown Threats via Known Backdoor Aggregation".

Stage I (this guide):   exploratory backdoor injection + backdoor aggregation
Stage II (reserved):    recovery fine-tuning
"""

from .cluster_loss import ClusterLoss

__all__ = ["ClusterLoss"]