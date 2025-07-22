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

def prob_to_prediction(inferred, num_preds):
    inferred_copy = inferred.clone()
    predictions = []

    for i in range(num_preds):
        temp_max = torch.argmax(inferred_copy)
        predictions.append(int(temp_max))
        inferred_copy[temp_max] = -inferred_copy[temp_max]
    predictions.sort()
    return predictions

def compare_multihot(multi_1, multi_2):
    cnt = 0
    for i in range(len(multi_1)):
        if multi_1[i] == multi_2[i]:
            cnt += 1
    return cnt / len(multi_1)

def outputs_to_label(out, size):
    res = []
    for i in out:
        i_pred = prob_to_prediction(i,size)
        res.append(i_pred)
    return res

def normalize(batch):
    for j in range(len(batch)):
        den = sum(batch[j])
        for i in range(len(batch[j])):
            batch[j][i] = batch[j][i]/den
    return batch

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

def train_grad_classifier(model, train_loader, test_loader, criterion, batch_size_pretrained_model, num_epochs=10, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history_training_accuracy = []
    history_test_accuracy = []
    history_training_loss = []
    history_test_loss = []

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
            
            y_batch = normalize(y_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

            # Accuracy
            outputs_labels = outputs_to_label(outputs, batch_size_pretrained_model)
            y_batch_labels = outputs_to_label(y_batch, batch_size_pretrained_model)
            for i in range(len(y_batch)):
                correct += compare_multihot(y_batch_labels[i],outputs_labels[i])

            total += y_batch.size(0)

        avg_loss = running_loss / total
        accuracy = 0
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}")

        _, test_accuracy, test_loss = evaluate_model(model, test_loader, criterion, batch_size_pretrained_model, device)

        history_training_accuracy.append(accuracy)
        history_test_accuracy.append(test_accuracy)
        history_training_loss.append(avg_loss)
        history_test_loss.append(test_loss)
    
    return history_training_accuracy, history_test_accuracy, history_training_loss, history_test_loss
        
def evaluate_model(model, test_loader, criterion, batch_size_pretrained_model, device='cpu'):
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    cosine_sims = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # normalize y_batch
            y_batch = normalize(y_batch)

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
            
            # Accuracy
            outputs_labels = outputs_to_label(outputs, batch_size_pretrained_model)
            y_batch_labels = outputs_to_label(y_batch, batch_size_pretrained_model)
            for i in range(len(y_batch)):
                correct += compare_multihot(y_batch_labels[i],outputs_labels[i])

            total += y_batch.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    avg_cosine_similarity = np.mean(cosine_sims)
    accuracy = 0
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f} | Cosine Similarity: {avg_cosine_similarity:.4f}, Test Accuracy: {accuracy:.4f}")

    return all_preds, accuracy, avg_loss

def state_dict_scale(sd, n):
    ret = {}
    for key in sd.keys():
        ret[key] = sd[key] * n
    return ret

