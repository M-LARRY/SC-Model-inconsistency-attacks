import copy
import torch
from model import Model
from torch.utils.data import DataLoader, Dataset

# Client update function (local training)
def local_train(model, dataloader, criterion, optimizer, epochs=1, device='cpu'):
    model.train()
    model.to(device)
    for _ in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# Server aggregation function (FedAvg)
def federated_avg(state_dicts):
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] += state_dicts[i][key]
        avg_state_dict[key] = avg_state_dict[key] / len(state_dicts)
    return avg_state_dict

# Simulate federated learning process
def federated_learning(clients_dataloaders, input_shape, num_classes, rounds=5, epochs=1, device='cpu'):
    # Initialize global model
    global_model = Model(input_shape, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    for round in range(rounds):
        print(f"Round {round + 1}")

        # Send global model to all clients, clients train locally
        local_state_dicts = []
        for dataloader in clients_dataloaders:
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
            local_state = local_train(local_model, dataloader, criterion, optimizer, epochs, device)
            local_state_dicts.append(local_state)

        # Aggregate updates from clients
        avg_state_dict = federated_avg(local_state_dicts)
        global_model.load_state_dict(avg_state_dict)

    return global_model

