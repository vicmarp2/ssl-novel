import torch
from torch.nn import functional as F

from .utils_pl import get_pair_indices

from .utils_pl import (bha_coeff, bha_coeff_loss)

from torch import Tensor
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PairLoss:
    def __init__(self, args):
        self.confidence_threshold = args.confidence_threshold
        self.similarity_threshold = args.similarity_threshold
        self.similarity_metric = bha_coeff
        self.distance_loss_metric = bha_coeff_loss

    def __call__(self, logits: Tensor, targets: Tensor, mode = 'unsupervised'):
        """
        Args:
            logits: (batch_size, num_classes) predictions of batch data
            targets: (batch_size, num_classes) one-hot labels
        Returns: Pair loss value as a Tensor.
        """
        if(mode == 'supervised'):
            targets_s = np.zeros(logits.size())
            targets_idx = [np.arange(targets_s.shape[0]), targets.tolist()]
            targets_s[targets_idx] = 1
            targets = torch.tensor(targets_s).to(device)
        indices = get_pair_indices(targets, ordered_pair=True)
        total_size = len(indices) // 2

        i_indices, j_indices = indices[:, 0], indices[:, 1]
        targets_max_prob = targets.max(dim=1).values
        targets_i_max_prob = targets_max_prob[i_indices]

        logits_j = logits[j_indices]
        targets_i = targets[i_indices]
        targets_j = targets[j_indices]

        # conf_mask should not track gradient
        conf_mask = (targets_i_max_prob > self.confidence_threshold).detach().float()

        similarities: Tensor = self.get_similarity(targets_i=targets_i,
                                                   targets_j=targets_j)
        # sim_mask should not track gradient
        sim_mask = F.threshold(similarities, self.similarity_threshold, 0).detach()

        distance = self.get_distance_loss(logits=logits_j,
                                          targets=targets_i)

        loss = conf_mask * sim_mask * distance

        loss = torch.sum(loss) / total_size

        return loss
       

    def get_similarity(self, targets_i, targets_j, *args):
        x, y = targets_i, targets_j
        return self.similarity_metric(targets_i, targets_j, *args)

    def get_distance_loss(self, logits, targets, *args):
        return self.distance_loss_metric(logits, targets, *args)