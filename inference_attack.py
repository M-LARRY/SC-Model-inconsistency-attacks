import torch
from torch import nn
import torch.nn.functional as F


def elementwise_diff_state_dicts(state_dict1, state_dict2, learning_rate):
    diff_dict = {}

    for key in state_dict1:
        if key in state_dict2:
            param1 = state_dict1[key]
            param2 = state_dict2[key]

            if param1.shape != param2.shape:
                raise ValueError(f"Shape mismatch for {key}: {param1.shape} vs {param2.shape}")

            diff_dict[key] = (param1 - param2) / learning_rate
        else:
            raise KeyError(f"Key {key} not found in second state_dict")

    return diff_dict

def generate_dataset():
    pass

def one_hot_encode(label, num_classes):
    encoding = torch.zeros(num_classes)
    encoding[label] = 1.0
    return encoding

def kl_div(p, q):
    # p, q are probability distributions (batch_size x classes)
    return nn.functional.kl_div(input=F.log_softmax(p, dim=1), target=q, reduction="batchmean")