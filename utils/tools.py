import random
from typing import Optional

import numpy as np
import torch
from sksurv.functions import StepFunction

def seed_everything(seed: int = 0) -> int:
    """
        Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def to_numpy(X):
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    return X

def to_value(X):
    if isinstance(X, torch.Tensor):
        X = X.item()
    return X

# def to_value(y):
#     assert type(y) is np.ndarray
#     if len(y.dtype) == 2:
#         # print(np.array([value for flag, value in y]))
#         return np.array([value for flag, value in y])
#     else:
#         return y

def set_value(y, y_value):
    y_ret = y.copy()
    y_ret[:] = [(y[i][0], y_value[i]) for i in range(y.shape[0])]
    return y_ret

def random_survival_function(fn, return_u=False):
    assert type(fn) is StepFunction
    u = np.random.uniform(0, 1)
    idx = np.argmax(fn.y <= u)
    t = fn.x[idx]
    if not return_u:
        return t
    else:
        return t, u

def get_nn_parameters(model):
    ret = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            ret[name] = param.data
    return ret

def is_point_in_sphere(point, center, radius2):
    dist2 = np.linalg.norm(point - center, ord=2)
    return dist2 <= radius2
