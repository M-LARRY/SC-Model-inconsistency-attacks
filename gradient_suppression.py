import copy
import torch
from model import Model1, Model2
from torch.utils.data import DataLoader, Dataset
from federated_learning import *
from utils import *

# Suppress ReLU layers
def suppress_relu(model, model_type):
    if model_type == 1:
        layers = ['conv1', 'conv2', 'conv3']
    elif model_type == 2:
        layers = ['conv1', 'conv2', 'fc1', 'fc2']
    else:
        print('Unknown model. Unable to suppress ReLU.')
    for layer_name in layers:
        layer = getattr(model, layer_name, None)
        if layer is not None:
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.data.zero_()
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.zero_()
    return model

# Simulate gradient suppression attack
def gradient_suppression(clients_dataloaders, input_shape, num_classes, trained_model_state_dict=None, target=0, epochs=1, device='cpu', model=1):
    # Initialize global model
    if model == 1:
        global_model = Model1(input_shape, num_classes).to(device)
    elif model == 2:
        global_model = Model2(input_shape, num_classes).to(device)
    else:
        print('Unknown model. Attack not started.')
        return None, None
    if trained_model_state_dict != None:
        global_model.load_state_dict(trained_model_state_dict)
    criterion = torch.nn.CrossEntropyLoss()

    # compute altered model weights
    altered_model = copy.deepcopy(global_model)
    altered_model = suppress_relu(altered_model, model_type=model)

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

    return global_model, local_state_dicts