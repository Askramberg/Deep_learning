import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import optuna
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt

# Define the U-Net model with batch normalization and dropout
class UNet(nn.Module):
    def __init__(self, num_layers, dropout_rate, num_filters):
        super(UNet, self).__init__()
        layers = [nn.Conv2d(1, num_filters, kernel_size=3, stride=1, padding=1),
                  nn.BatchNorm2d(num_filters),
                  nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Conv2d(num_filters, 3, kernel_size=3, stride=1, padding=1))  # Output 3 channels for 3 classes
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Custom Dataset class for handling data loading and preprocessing
class CustomDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.images = os.listdir(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        match = re.search(r'\d+', img_name)
        if match:
            img_number = match.group()
            label_name = f"slice__{int(img_number):03d}.tif"
        else:
            raise ValueError(f"Cannot extract number from {img_name}")

        img_path = os.path.join(self.data_path, img_name)
        label_path = os.path.join(self.label_path, label_name)

        image = Image.open(img_path)
        label = Image.open(label_path)

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Convert image tensor to float type
        image = image.float()

        # Process label
        np_label = np.array(label, dtype=np.float32)
        np_label = np.round(np_label / 65535.0 * 255)  # Scale label to 8-bit
        label_tensor = torch.zeros(np_label.shape, dtype=torch.long)
        grayscale_to_class_mapping = {0: 2, 128: 1, 255: 0}
        for grayscale_value, class_id in grayscale_to_class_mapping.items():
            mask = np_label == grayscale_value
            label_tensor[mask] = class_id

        return image, label_tensor

# Transformations with Data Augmentation (applied to images only)
image_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

# Load the dataset (small subset)
data_path = r"/zhome/4c/e/181174/training_dataset/data"
label_path = r"/zhome/4c/e/181174/training_dataset/labels"


full_dataset = CustomDataset(data_path, label_path, transform=image_transform)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Define the objective function for hyperparameter optimization
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers = trial.suggest_categorical('num_layers', [2, 3, 4, 5])  # Including friend's likely configuration
    learning_rate = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3, 0.001])  # Including friend's configuration
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.3, 0.5])  # Adjust based on friend's model
    num_epochs = trial.suggest_categorical('num_epochs', [10, 20, 30, 40, 50])  # Including friend's configuration
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])  # Including friend's configuration
    num_filters = trial.suggest_categorical('num_filters', [16, 32, 64, 128])  # Adjust based on friend's initial layer

    model = UNet(num_layers=num_layers, dropout_rate=dropout_rate, num_filters=num_filters).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and Validation loop with loss tracking
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Plot and save training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"loss_curve_trial_exp2_{trial.number}.png")
    plt.close()

    return val_loss

# Hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # Adjust number of trials as needed

# Best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Save the best hyperparameters to a file for reference
best_hyperparameters = study.best_params
with open("best_hyperparameters_2.txt", "w") as f:
    for key, value in best_hyperparameters.items():
        f.write(f"{key}: {value}\n")


