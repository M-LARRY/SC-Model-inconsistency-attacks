import copy
import torch
from model import Model1, Model2
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
def federated_learning(clients_dataloaders, input_shape, num_classes, criterion=torch.nn.CrossEntropyLoss(), lr=0.001, rounds=5, epochs=1, device='cpu', model=1):
    # Initialize global model
    if model == 1:
        global_model = Model1(input_shape, num_classes).to(device)
    elif model == 2:
        global_model = Model2(input_shape, num_classes).to(device)
    else:
        print('Unknown model. Simulation not started.')
        return None, None

    for round in range(rounds):
        print(f"Round {round + 1}")

        # Send global model to all clients, clients train locally
        local_state_dicts = []
        for dataloader in clients_dataloaders:
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
            local_state = local_train(local_model, dataloader, criterion, optimizer, epochs, device)
            local_state_dicts.append(local_state)

        # Aggregate updates from clients
        avg_state_dict = federated_avg(local_state_dicts)
        global_model.load_state_dict(avg_state_dict)

    return global_model, local_state_dicts

# for debug usage
def local_train_debug(model, dataloader, criterion, learning_rate, device='cpu'):
    model.train()
    model.to(device)

    # Use plain SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Clone initial state for gradient estimation
    initial_state = copy.deepcopy(model.state_dict())
    torch.save(initial_state, 'DEBUG_initial_state.pth')

    # === Step 1: Get one batch ===
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)

    # === Step 2: Compute true gradients ===
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    true_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            true_grads[name] = param.grad.detach().clone()

    torch.save(true_grads, 'DEBUG_true_grads.pth')

    # === Step 3: Optimizer step ===
    optimizer.step()

    # === Step 4: Save updated state for gradient estimation ===
    updated_state = model.state_dict()
    torch.save(updated_state, 'DEBUG_updated_state.pth')

    # === Step 5: Estimate gradients using weight difference ===
    estimated_grads = {}
    for key in initial_state:
        if key in updated_state:
            p_t = initial_state[key]
            p_t1 = updated_state[key]
            if torch.is_tensor(p_t) and p_t.dtype.is_floating_point:
                estimated_grads[key] = (p_t - p_t1) / learning_rate

    # === Step 6: Compare gradients ===
    print("\nGradient comparison (L2 norm of difference):")
    for name, real_grad in true_grads.items():
        if name in estimated_grads:
            est_grad = estimated_grads[name]
            if est_grad.shape == real_grad.shape:
                diff = torch.norm(real_grad - est_grad).item()
                print(f"{name}: L2 diff = {diff:.6e}")
            else:
                print(f"{name}: shape mismatch ({real_grad.shape} vs {est_grad.shape})")
        else:
            print(f"{name}: not found in estimated gradients")

    return updated_state