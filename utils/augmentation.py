import torch
import torch.nn.functional as F 
import numpy as np


def cutout(x, ratio=.1):
    """
    Parameters:
    x: input sequence, # dims: [n_batches, seq_len, feature_dim]
    ratio: ratio for cutout. Default: 0.1
    Returns: 
    x after random cutout
    Paper: Improved regularization of convolutional neural networks with cutout. https://arxiv.org/abs/1708.04552
    """
    seq_len = x.shape[1]
    win_len = int(np.ceil(seq_len * ratio))
    # randomly select a subsequence of given length
    start = np.random.randint(0, seq_len - win_len + 1) 
    end = start + win_len
    # output = x.copy()
    # output[:, start:end, :] = 0
    mask = torch.ones_like(x)
    mask[:, start:end, :] = 0
    # return np.squeeze(output, axis=0)
    return x * mask


def mixup_data(x, y, alpha=0.4):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_layer(y, device, alpha = 0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.shape[0]
    indices_perm = torch.randperm(batch_size)
    if device.type != 'cpu':
        indices_perm.cuda()
    y_a, y_b = y, y[indices_perm]
    
    return indices_perm, y_a, y_b, lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    """
    Define the loss (usually weighted cross entropy) for Mixup augmentation.
    Parameters:
    criterion: loss function.
    label: predictions.
    y_a, y_b: labels generated from mixup() method above.
    lam: lambda, the mixing factor.
    Returns:
    mixup loss for the batch.
    """
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

def cutmix(x, y, alpha=0.4):
    '''Cutmix augmentation. Return mixed inputs, pairs of targets, and lambda'''

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    indices_perm = torch.randperm(batch_size).cuda()
    win_len = int(np.ceil(seq_len * lam))
    start = np.random.randint(0, seq_len - win_len + 1) 
    end = start + win_len
    mixed_x = x
    mixed_x[:, start:end, :] = x[indices_perm, start:end, :]
    lam = 1. - win_len / seq_len
    y_a, y_b = y, y[indices_perm]
    return mixed_x, y_a, y_b, lam

