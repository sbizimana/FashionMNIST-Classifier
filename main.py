import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tf
import os
import pickle
import matplotlib.pyplot as plt
from plotcm import plot_confusion_matrix

# Use GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import FashionMNIST data
TRAINSET = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=tf.Compose([tf.ToTensor()])
)
TESTSET = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=False,
    download=True,
    transform=tf.Compose([tf.ToTensor()])
)

# Create training and testing data loaders
TRAINLOADER = DataLoader(TRAINSET, batch_size=100)
TESTLOADER = DataLoader(TESTSET, batch_size=100)


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=28, kernel_size=3),
            nn.BatchNorm2d(28),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=56, kernel_size=3),
            nn.BatchNorm2d(56),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=112, kernel_size=3),
            nn.BatchNorm2d(112),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=224, kernel_size=3),
            nn.BatchNorm2d(224),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.size = 10
        self.fc1 = nn.Linear(224 * self.size * self.size, 512)
        self.drop1 = nn.Dropout2d(0.33)
        self.fc2 = nn.Linear(512, 128)
        self.drop2 = nn.Dropout2d(0.25)
        self.fc3 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout2d(0.1)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.reshape(-1, 224 * self.size * self.size)

        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.out(x)
        return x


def train(model, epochs, optimizer, loss_function, device=DEVICE, train_loader=TRAINLOADER, test_loader=TESTLOADER, testing_mod=500):
    count = 0
    best_accuracy = 0
    accuracies = []
    losses = []
    iterations = []
    interrupted = False
    try:
        print(f"Training using device: {device}")
        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(device)

                outputs = model(images)
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count += 1

                if count % testing_mod == 0:
                    total = 0
                    correct = 0
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        predictions = torch.max(outputs, 1)[1].to(device)
                        correct += (predictions == labels).sum()
                        total += len(labels)

                    accuracy = correct / total * 100
                    accuracies.append(accuracy.item())
                    losses.append(loss.item())
                    iterations.append(count)

                    if accuracy.item() > best_accuracy:
                        best_accuracy = accuracy.item()
                        torch.save(model.state_dict(), "./models/FMNIST_model_best.pth")

                    print(f"Epoch: {epoch + 1}/{epochs}, Iteration: {count}, Loss: {loss.item()}, Accuracy: {round(accuracy.item(), 2)}%")

            if epoch == epochs - 1:
                print(f"Epoch: {epoch + 1}/{epochs}, Iteration: {count}, Loss: {loss.item()}, Accuracy: {round(accuracy.item(), 2)}%")

    except KeyboardInterrupt:
        interrupted = True
    if interrupted:
        print(f"{count} completed iterations...")

    torch.save(model.state_dict(), f"./models/FMNIST_model_last.pth")
    return model, accuracies, losses, iterations


# Plot the confusion matrix
def confusion_matrix(predictions, labels, classes, title="Confusion Matrix", device=DEVICE):
    labels = labels.to(device)
    predictions = predictions.to(device)
    stacked = torch.stack((labels, predictions.argmax(dim=1)), dim=1)
    confusion_mtx = torch.zeros(10, 10, dtype=torch.int64)

    for prediction in stacked:
        t, p = prediction.tolist()
        confusion_mtx[int(t), int(p)] = confusion_mtx[int(t), int(p)] + 1

    plot_confusion_matrix(confusion_mtx, classes, title=title)


# Get each prediction from a model
def get_predictions(model, loader, device=DEVICE):
    with torch.no_grad():
        all_predictions = torch.tensor([]).to(device)

        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            preds = model(images)
            all_predictions = torch.cat((all_predictions, preds), dim=0)
        return all_predictions


if __name__ == "__main__":
    # Initialize hyperparameters
    lr = 0.001
    epochs = 50

    model = NN()
    model.to(DEVICE)

    # Train or load model
    file = "./models/FMNIST_model_best.pth"
    file2 = "./models/FMNIST_model_last.pth"
    if os.path.exists(file) and os.path.exists(file2):
        model.load_state_dict(torch.load(file))
        with open("./data/lists.plk", "rb") as f:
            accuracies, losses, iterations = pickle.load(f)
        print(f"Best Accuracy: {round(max(accuracies), 2)}%, Best Loss: {min(losses)}")
    else:
        loss_function = nn.CrossEntropyLoss()
        optimizer1 = optim.Adam(model.parameters(), lr=lr)
        optimizer2 = optim.Adagrad(model.parameters(), lr=lr, lr_decay=1e-2)
        model, accuracies, losses, iterations = train(model, epochs, optimizer1, loss_function)
        with open("./data/lists.plk", "wb") as f:
            pickle.dump([accuracies, losses, iterations], f)
        print(f"Best Accuracy: {round(max(accuracies), 2)}%, Best Loss: {min(losses)}")


    preds_loader = DataLoader(TESTSET, batch_size=10000, shuffle=False)
    all_predictions = get_predictions(model.to("cpu").eval(), preds_loader, device="cpu")
    plt.rcParams['figure.figsize'] = [10, 10]
    confusion_matrix(all_predictions, TESTSET.targets, TESTSET.classes, title="Confusion Matrix")
    plt.show()
















