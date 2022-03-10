# credits to
# https://github.com/zijian-hu/SimPLE/
import torch
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader, Subset
from model.wrn import WideResNet
import torch.nn.functional as F
from torch.utils.data import DataLoader

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])



def get_similarity(targets_i, targets_j, *args):
    x, y = targets_i, targets_j
    return bha_coeff(targets_i, targets_j, *args)

def get_distance_loss(logits, targets, *args):
    return bha_coeff_loss(logits, targets, *args)

def get_pair_indices(inputs, ordered_pair = False):
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


def accuracy(output, target, topk=(1,)):
    """
    Function taken from pytorch examples:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def alpha_weight(alpha, t1, t2, curr_epoch):
    """ Calculate alpha regulariser
    """
    if curr_epoch < t1:
        return 0.0
    elif curr_epoch > t2:
        return alpha
    else:
        return ((curr_epoch-t1) / (t2-t1))*alpha


def plot(metric, label, color='b'):
    """  Generates a plot of a given metric given along the epochs
    """
    epochs = range(len(metric))
    plt.plot(epochs, metric, color, label=label)
    plt.title(label)
    plt.xticks(np.arange(0, len(epochs), 2.0))
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.show()


def plot_model_attr(modelpath, attrname, label, color='b'):
    """ Generates a plot of a given attribute from a model
        Training, validation, test loss
    """
    model_cp = torch.load(pjoin(modelpath))
    plot(model_cp[attrname], label, color)

def plot_model(modelpath):
    """ Generates a plot of training and validation loss
    """
    model_cp = torch.load(pjoin(modelpath))
    plt.plot(model_cp['training_losses'])
    plt.plot(model_cp['validation_losses'])
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


def validation_set(base_dataset, num_validation, num_classes):
    '''
    args: 
        base_dataset : (torch.utils.data.Dataset)
    returns : (torch.utils.data.Dataset) subset 
    Description:
        This function samples even ammount of images from each class
        from the base dataset given up to the size of the validation dataset
    '''
    labels = base_dataset.targets
    label_per_class = num_validation // num_classes
    labels = np.array(labels)
    validation_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        validation_idx.extend(idx)
    validation_idx = np.array(validation_idx)
    np.random.shuffle(validation_idx)
    assert len(validation_idx) == num_validation
    return Subset(base_dataset, validation_idx)


def test_accuracy(testdataset, filepath = "./path/to/model.pth.tar"):
    # CREATE LOADER 
   
    test_loader = DataLoader(testdataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=1)
    
    
    # RETRIEVE MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelpath = torch.load(pjoin(filepath))
    model = WideResNet(modelpath['model_depth'],
                       modelpath['num_classes'], widen_factor=modelpath['model_width'], dropRate=modelpath['drop_rate'])
    model = model.to(device)
    model.load_state_dict(modelpath['model_state_dict'])

    # CALCULATE ACCURACY
    model.eval()
    total_accuracy = []
    for x_test, y_test in test_loader:
        with torch.no_grad():
            x_test, y_test = x_test.to(device), y_test.to(device)
            output_test = model(x_test)
            acc = accuracy(output_test, y_test)
            total_accuracy.append(sum(acc))
    print('Accuracy of the network on test images: %d %%' % (
                sum(total_accuracy)/len(total_accuracy)))

 
def test_error(testdataset, filepath = "./path/to/model.pth.tar"):
    # CREATE LOADER 

    test_loader = DataLoader(testdataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=1)


    # RETRIEVE MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelpath = torch.load(pjoin(filepath))
    model = WideResNet(modelpath['model_depth'],
                    modelpath['num_classes'], widen_factor=modelpath['model_width'], dropRate=modelpath['drop_rate'])
    model = model.to(device)
    model.load_state_dict(modelpath['model_state_dict'])

    # CALCULATE ACCURACY
    model.eval()
    total_accuracy = []
    for x_test, y_test in test_loader:
        with torch.no_grad():
            x_test, y_test = x_test.to(device), y_test.to(device)
            output_test = model(x_test)
            acc = accuracy(output_test, y_test)
            total_accuracy.append(sum(acc))
    acc = float((sum(total_accuracy) / len(total_accuracy)))
    err = (1 - acc * 0.01) * 100
    print('Error of the network on test images: {:.2f} %'.format(err))
