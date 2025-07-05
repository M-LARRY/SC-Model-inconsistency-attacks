import copy
import torch
from model import Model
from torch.utils.data import DataLoader, Dataset
from federated_learning import *
from utils import *

# Suppress ReLU layers
def suppress_relu(model):
    for layer_name in ['conv1', 'conv2', 'conv3']:
        layer = getattr(model, layer_name, None)
        if layer is not None:
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.data.zero_()
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.zero_()
    return model

# Simulate gradient suppression attack
def gradient_suppression(clients_dataloaders, input_shape, num_classes, target=0, epochs=1, device='cpu'):
    # Initialize global model
    global_model = Model(input_shape, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # compute altered model weights
    altered_model = copy.deepcopy(global_model)
    altered_model = suppress_relu(altered_model)

    # Send global model to target, altered model to remainig clients. clients train locally
    local_state_dicts = []
    for dataloader in clients_dataloaders:
        if dataloader == clients_dataloaders[target]:
            print("target is client", target)
            local_model = copy.deepcopy(global_model)
        else:
            local_model = copy.deepcopy(altered_model)
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
        local_state = local_train(local_model, dataloader, criterion, optimizer, epochs, device)
        local_state_dicts.append(local_state)

    # Aggregate updates from clients
    avg_state_dict = federated_avg(local_state_dicts)
    global_model.load_state_dict(avg_state_dict)

    mse = state_dicts_mse(local_state_dicts[target], avg_state_dict)
    frac_equal = state_dicts_fraction_equal_params(local_state_dicts[target], avg_state_dict)
    avg_cos_sim = state_dicts_average_cosine_similarity(local_state_dicts[target], avg_state_dict)

    print("Comparing target update with global update:")
    print(f"Average MSE: {mse:.4f}")
    print(f"Fraction of exactly equal params: {frac_equal:.4f}")
    print(f"Average cosine similarity: {avg_cos_sim:.4f}")

    return global_model, avg_state_dict, local_state_dicts