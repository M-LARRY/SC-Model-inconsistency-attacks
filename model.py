import torch
import torch.nn as nn
import torch.nn.functional as F

class Model1(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.conv1(dummy_input)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)
            self.flattened_size = x.numel()

        self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        y = F.softmax(x, dim=1)
        return y

class Model2(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Model2, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Halves H and W each time

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y = F.softmax(x, dim=1)               # Return class probabilities
        return y


class GradientClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=[512, 256], output_dim=10):
        super(GradientClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x