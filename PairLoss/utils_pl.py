import torch
from torch.nn import functional as F
# for type hint
from torch import Tensor


def get_pair_indices(inputs: Tensor, ordered_pair: bool = False) -> Tensor:
    """
    Get pair indices between each element in input tensor
    Args:
        inputs: input tensor
        ordered_pair: if True, will return ordered pairs. (e.g. both inputs[i,j] and inputs[j,i] are included)
    Returns: a tensor of shape (K, 2) where K = choose(len(inputs),2) if ordered_pair is False.
        Else K = 2 * choose(len(inputs),2). Each row corresponds to two indices in inputs.
    """
    indices = torch.combinations(torch.tensor(range(len(inputs))), r=2)

    if ordered_pair:
        # make pairs ordered (e.g. both (0,1) and (1,0) are included)
        indices = torch.cat((indices, indices[:, [1, 0]]), dim=0)

    return indices

import torch

# for type hint
from typing import Union, Optional, Sequence, Callable
from torch import Tensor

ScalarType = Union[int, float, bool]


def reduce_tensor(inputs: Tensor, reduction: str) -> Tensor:
    if reduction == 'mean':
        return torch.mean(inputs)

    elif reduction == 'sum':
        return torch.sum(inputs)

    return inputs


def to_tensor(data: Union[ScalarType, Sequence[ScalarType]],
              dtype: Optional[torch.dtype] = None,
              device: Optional[Union[torch.device, str]] = None,
              tensor_like: Optional[Tensor] = None) -> Tensor:
    if tensor_like is not None:
        dtype = tensor_like.dtype if dtype is None else dtype
        device = tensor_like.device if device is None else device

    return torch.tensor(data, dtype=dtype, device=device)


def bha_coeff_log_prob(log_p: Tensor, log_q: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of log(p) and log(q); the more similar the larger the coefficient
    :param log_p: (batch_size, num_classes) first log prob distribution
    :param log_q: (batch_size, num_classes) second log prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none"
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    # numerical unstable version
    # coefficient = torch.sum(torch.sqrt(p * q), dim=dim)
    # numerical stable version
    coefficient = torch.sum(torch.exp((log_p + log_q) / 2), dim=dim)

    return reduce_tensor(coefficient, reduction)


def bha_coeff(p: Tensor, q: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of p and q; the more similar the larger the coefficient
    :param p: (batch_size, num_classes) first prob distribution
    :param q: (batch_size, num_classes) second prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none"
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    log_p = torch.log(p)
    log_q = torch.log(q)

    return bha_coeff_log_prob(log_p, log_q, dim=dim, reduction=reduction)

def bha_coeff_loss(logits: Tensor, targets: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of p and q; the more similar the larger the coefficient
    :param logits: (batch_size, num_classes) model predictions of the data
    :param targets: (batch_size, num_classes) label prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    log_probs = F.log_softmax(logits, dim=dim)
    log_targets = torch.log(targets)

    # since BC(P,Q) is maximized when P and Q are the same, we minimize 1 - B(P,Q)
    return 1. - bha_coeff_log_prob(log_probs, log_targets, dim=dim, reduction=reduction)
