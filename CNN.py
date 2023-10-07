import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split


class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        if in_channels == 1:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
            # Input: 28x28 -> Conv1 -> 26x26 -> MaxPool -> 13x13
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
            # 13x13 -> Conv2 -> 11x11 -> MaxPool -> 5x5
            self.fc1 = nn.Linear(16 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
        elif in_channels == 3:
            self.conv1 = nn.Conv2d(3, 24, kernel_size=3)
            # Input: 32x32 -> Conv1 -> 30x30 -> MaxPool -> 15x15
            self.conv2 = nn.Conv2d(24, 44, kernel_size=3)
            # 15x15 -> Conv2 -> 13x13 -> MaxPool -> 6x6
            self.fc1 = nn.Linear(44*6*6, 256)  # Adjusted input size for fc1
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))  # Use F.tanh as the activation
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))  # Use F.relu as the activation
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.selu(self.fc1(x))  # Use F.selu as the activation
        x = F.relu(self.fc2(x))  # Use F.relu as the activation
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def load_dataset(
        dataset_name: str,
):
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset, valid_dataset = random_split(
            full_dataset, [48000, 12000])

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset, valid_dataset = random_split(
            full_dataset, [38000, 12000])

    else:
        raise Exception("Unsupported dataset.")

    return train_dataset, valid_dataset


class L2_Regularization(nn.Module):
    def __init__(self, model, lamda):
        super(L2_Regularization, self).__init__()
        self.model = model
        self.lamda = lamda

    def forward(self):
        l2_reg = torch.tensor(0.0)
        for parameter in self.model.parameters():
            l2_reg += torch.norm(parameter, p=2) ** 2
        return self.lamda * l2_reg


def train(
        model,
        train_dataset,
        valid_dataset,
        device,
        n_epochs=10,
        learning_rate=1e-3,
        batch_size=200,
        lamda=.00001
):
    # Make sure to fill in the batch size.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    l2_reg = L2_Regularization(model, lamda=lamda).to(device)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss += l2_reg()  # Apply L2 regularization
            loss.backward()
            optimizer.step()

        if epoch % 3 == 0:
            model.eval()
            total_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    loss = loss_function(output, target)
                    total_loss += loss.item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum().item()

            loss = total_loss / len(valid_loader.dataset)
            acc = 100.0 * correct / len(valid_loader.dataset)
            print(
                f'Epoch {epoch}: Validation Loss={loss:.4f}, Accuracy={acc:.2f}%')

    results = dict(
        model=model
    )

    return results


def CNN(dataset_name, device):

    # CIFAR-10 has 3 channels whereas MNIST has 1.
    if dataset_name == "CIFAR10":
        in_channels = 3
    elif dataset_name == "MNIST":
        in_channels = 1
    else:
        raise AssertionError(f'invalid dataset: {dataset_name}')

    model = Net(in_channels).to(device)

    train_dataset, valid_dataset = load_dataset(dataset_name)

    results = train(model, train_dataset, valid_dataset, device)

    return results
