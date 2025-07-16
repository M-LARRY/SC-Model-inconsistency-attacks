import torch
import torch.nn.functional as F

def dict_to_tensor(dict):
    tensor = []
    for key in dict.keys():
        flattened_layer = dict[key].flatten()
        tensor.extend(flattened_layer)
    return torch.Tensor(tensor)

def state_dicts_mse(sd1, sd2):
    mse_total = 0.0
    count = 0
    for k in sd1.keys():
        if k in sd2:
            diff = sd1[k] - sd2[k]
            mse_total += (diff**2).mean().item()
            count += 1
    return mse_total / count if count > 0 else None

def state_dicts_fraction_equal_params(sd1, sd2):
    equal_count = 0
    total_count = 0
    for k in sd1.keys():
        if k in sd2:
            equal_count += torch.sum(sd1[k] == sd2[k]).item()
            total_count += sd1[k].numel()
    return equal_count / total_count if total_count > 0 else None

def state_dicts_average_cosine_similarity(sd1, sd2):
    cos_sims = []
    for k in sd1.keys():
        if k in sd2:
            v1 = sd1[k].flatten()
            v2 = sd2[k].flatten()
            cos_sim = F.cosine_similarity(v1, v2, dim=0).item()
            cos_sims.append(cos_sim)
    return sum(cos_sims) / len(cos_sims) if cos_sims else None

def state_dicts_cosine_similarity(sd1, sd2):
    for key in sd1.keys():
        if key in sd2.keys():
            cs = F.cosine_similarity(sd1[key].flatten(), sd2[key].flatten(), dim=0)
            print(cs)
    return

def state_dicts_mse_per_layer(sd1, sd2):
    for key in sd1.keys():
        if key in sd2.keys():
            mse = F.mse_loss(sd1[key].flatten(), sd2[key].flatten())
            print(mse)
    return

def compare_state_dicts(sd1, sd2):
    sd1 = dict_to_tensor(sd1)
    sd2 = dict_to_tensor(sd2)
    return F.cosine_similarity(sd1, sd2, dim=0)