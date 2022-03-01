import torch
from torch.nn import functional as F

from .utils_pl import get_pair_indices, reduce_tensor, bha_coeff_log_prob, l2_distance

from typing import Union, Dict
from .utils_pl import (bha_coeff, bha_coeff_distance, l2_distance, softmax_cross_entropy_loss, l2_dist_loss, bha_coeff_loss)

# for type hint
from typing import Optional
from torch import Tensor

SimilarityType = Union[bha_coeff]
DistanceType = Union[bha_coeff_distance, l2_distance]
DistanceLossType = Union[softmax_cross_entropy_loss, l2_dist_loss, bha_coeff_loss]


class SupervisedLoss:
    def __init__(self, reduction: str = 'mean'):
        self.loss_use_prob = False
        self.loss_fn = softmax_cross_entropy_loss

        self.reduction = reduction

    def __call__(self, logits: Tensor, probs: Tensor, targets: Tensor) -> Tensor:
        loss_input = probs if self.loss_use_prob else logits
        loss = self.loss_fn(loss_input, targets, dim=1, reduction=self.reduction)

        return loss


class UnsupervisedLoss:
    def __init__(self,
                 loss_type: str,
                 loss_thresholded: bool = False,
                 confidence_threshold: float = 0.,
                 reduction: str = "mean"):
        if loss_type in ["entropy", "cross entropy"]:
            self.loss_use_prob = False
            self.loss_fn = softmax_cross_entropy_loss
        self.loss_thresholded = loss_thresholded
        self.confidence_threshold = confidence_threshold
        self.reduction = reduction

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        loss = self.loss_fn(logits, targets, dim=1, reduction="none")

        if self.loss_thresholded:
            targets_mask = (targets.max(dim=1).values > self.confidence_threshold)

            if len(loss.shape) > 1:
                # mse_loss returns a matrix, need to reshape mask
                targets_mask = targets_mask.view(-1, 1)

            loss *= targets_mask.float()

        return reduce_tensor(loss, reduction=self.reduction)

class PairLoss:
    def __init__(self,
                 similarity_metric: SimilarityType,
                 distance_loss_metric: DistanceLossType,
                 confidence_threshold: float,
                 similarity_threshold: float,
                 similarity_type: str,
                 distance_loss_type: str,
                 reduction: str = "mean"):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold

        self.similarity_type = similarity_type
        self.distance_loss_type = distance_loss_type

        self.reduction = reduction

        self.similarity_metric = similarity_metric
        self.distance_loss_metric = distance_loss_metric

    def __call__(self,
                 logits: Tensor,
                 targets: Tensor,
                 *args,
                 indices: Optional[Tensor] = None,
                 **kwargs) -> Tensor:
        """
        Args:
            logits: (batch_size, num_classes) predictions of batch data
            probs: (batch_size, num_classes) softmax probs logits
            targets: (batch_size, num_classes) one-hot labels
        Returns: Pair loss value as a Tensor.
        """
        if indices is None:
            indices = get_pair_indices(targets, ordered_pair=True)
        total_size = len(indices) // 2

        i_indices, j_indices = indices[:, 0], indices[:, 1]
        targets_max_prob = targets.max(dim=1).values

        return self.compute_loss(logits_j=logits[j_indices],
                                 targets_i=targets[i_indices],
                                 targets_j=targets[j_indices],
                                 targets_i_max_prob=targets_max_prob[i_indices],
                                 total_size=total_size)

    def compute_loss(self,
                     logits_j: Tensor,
                     targets_i: Tensor,
                     targets_j: Tensor,
                     targets_i_max_prob: Tensor,
                     total_size: int):
        # conf_mask should not track gradient
        conf_mask = (targets_i_max_prob > self.confidence_threshold).detach().float()

        similarities: Tensor = self.get_similarity(targets_i=targets_i,
                                                   targets_j=targets_j,
                                                   dim=1)
        # sim_mask should not track gradient
        sim_mask = F.threshold(similarities, self.similarity_threshold, 0).detach()

        distance = self.get_distance_loss(logits=logits_j,
                                          targets=targets_i,
                                          dim=1,
                                          reduction='none')

        loss = conf_mask * sim_mask * distance

        if self.reduction == "mean":
            loss = torch.sum(loss) / total_size
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss

    def get_similarity(self, targets_i, targets_j, *args):
        x, y = targets_i, targets_j
        return self.similarity_metric(targets_i, targets_j, *args)

    def get_distance_loss(self, logits, targets, *args):
        return self.distance_loss_metric(logits, targets, *args)