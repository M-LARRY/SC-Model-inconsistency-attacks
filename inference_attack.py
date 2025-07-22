import torch
import torch.nn.functional as F
import numpy as np
from utils import *
from torch import nn
from numpy.linalg import norm

def remove_layers(sd, layers_to_remove):
    res = {}
    for key in sd.keys():
        if key not in layers_to_remove:
            res[key] = sd[key]
    return res

def estimate_gradient(state_dict1, state_dict2, learning_rate):
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

def one_hot_encode(label, num_classes):
    encoding = torch.zeros(num_classes)
    encoding[label] = 1.0
    return encoding

def multi_hot_encode(label, num_classes):
    encoding = torch.zeros(num_classes)
    for x in label:
        encoding[x.item()] = 1.0
    return encoding

def kl_div(p, q):
    # p, q are probability distributions (batch_size x classes)
    return nn.functional.kl_div(input=F.log_softmax(p, dim=1), target=q, reduction="batchmean")

def train_grad_classifier(model, train_loader, test_loader, criterion, num_epochs=10, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)  # raw logits

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

            # Accuracy
            print(outputs)
            #preds = torch.argmax(outputs, dim=1)
            #correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = running_loss / total
        accuracy = 0
        #accuracy = correct / total * 100

        # print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        evaluate_model(model, test_loader, criterion, device)
        
def evaluate_model(model, test_loader, criterion, device='cuda'):
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_preds = []
    all_labels = []
    cosine_sims = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)  # sigmoid output
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            preds = outputs.cpu().numpy()
            labels = y_batch.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

            # Compute cosine similarity per sample
            for p, l in zip(preds, labels):
                # Avoid division by zero
                if norm(p) == 0 or norm(l) == 0:
                    cosine_sims.append(0.0)
                else:
                    cos_sim = np.dot(p, l) / (norm(p) * norm(l))
                    cosine_sims.append(cos_sim)

    avg_loss = total_loss / len(test_loader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    avg_cosine_similarity = np.mean(cosine_sims)

    print(f"Test Loss: {avg_loss:.4f} | Cosine Similarity: {avg_cosine_similarity:.4f}")

    return all_preds

def state_dict_scale(sd, n):
    ret = {}
    for key in sd.keys():
        ret[key] = sd[key] * n
    return ret

