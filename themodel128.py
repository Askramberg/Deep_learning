import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import pickle

class CustomDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        # Filter out non-image files
        valid_extensions = ('.tiff', '.tif')
        self.image_paths = [path for path in image_paths if path.lower().endswith(valid_extensions)]
        self.label_paths = [path for path in label_paths if path.lower().endswith(valid_extensions)]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_paths[idx])
        np_image = np.array(image, dtype=np.float32)
        # Normalize the image
        normalized_image = np_image / 65535.0  # For 16-bit images

        # Load and process the label data
        label_image = Image.open(self.label_paths[idx])
        label_array = np.array(label_image, dtype=np.float32)

        grayscale_to_class_mapping = {0: 0, 128: 1, 255: 2}

        # Map grayscale values to class labels
        mapped_labels = np.copy(label_array)
        for grayscale_value, class_id in grayscale_to_class_mapping.items():
            mapped_labels[label_array == grayscale_value] = class_id

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(normalized_image).unsqueeze(0)
        label_tensor = torch.from_numpy(mapped_labels)

        return image_tensor, label_tensor

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Define a helper function for creating a block
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )

        # Encoder
        self.e11 = conv_block(1, 64)
        self.e12 = conv_block(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = conv_block(64, 128)
        self.e22 = conv_block(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = conv_block(128, 256)
        self.e32 = conv_block(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = conv_block(256, 512)
        self.e42 = conv_block(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = conv_block(512, 1024)
        self.e52 = conv_block(1024, 1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = conv_block(1024, 512)
        self.d12 = conv_block(512, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = conv_block(512, 256)
        self.d22 = conv_block(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = conv_block(256, 128)
        self.d32 = conv_block(128, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = conv_block(128, 64)
        self.d42 = conv_block(64, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = self.e11(x)
        xe12 = self.e12(xe11)
        xp1 = self.pool1(xe12)

        xe21 = self.e21(xp1)
        xe22 = self.e22(xe21)
        xp2 = self.pool2(xe22)

        xe31 = self.e31(xp2)
        xe32 = self.e32(xe31)
        xp3 = self.pool3(xe32)

        xe41 = self.e41(xp3)
        xe42 = self.e42(xe41)
        xp4 = self.pool4(xe42)

        xe51 = self.e51(xp4)
        xe52 = self.e52(xe51)

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.d11(xu11)
        xd12 = self.d12(xd11)

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.d21(xu22)
        xd22 = self.d22(xd21)

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.d31(xu33)
        xd32 = self.d32(xd31)

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.d41(xu44)
        xd42 = self.d42(xd41)

        # Output layer
        out = self.outconv(xd42)

        return out

def get_image_paths(data_dir, label_dir):
    data_paths = [os.path.join(data_dir, img) for img in sorted(os.listdir(data_dir))]
    label_paths = [os.path.join(label_dir, lbl) for lbl in sorted(os.listdir(label_dir))]
    return data_paths, label_paths


from torch.utils.data import DataLoader, random_split
from torch import optim
import os

### STEP 1: ###
### Define your dataset paths ###

data_dir_train = '/zhome/10/0/181468/xray/data_128_128' # PUT IN YOUR DIRECTORY PATH FOR ALL THESE
label_dir_train = '/zhome/10/0/181468/xray/label_128_128'
image_paths_train, label_paths_train = get_image_paths(data_dir_train, label_dir_train)
# Validation folder
data_dir_val = '/zhome/10/0/181468/xray/VAL_DATA_50'
label_dir_val = '/zhome/10/0/181468/xray/VAL_LABEL_50'
image_paths_val, label_paths_val = get_image_paths(data_dir_val, label_dir_val)

### Get image paths and create the full dataset ###
# Training loader
dataset_train = CustomDataset(image_paths=image_paths_train, label_paths=label_paths_train)
train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
# Validation loader
dataset_val = CustomDataset(image_paths=image_paths_val, label_paths=label_paths_val)
val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False)

# Step 2: Model Initialization
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
model = UNet(n_class=3)
model.to(device)

# Step 3: Loss Function & Optimizer
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 4: Training Loop
num_epochs = 100  # ADJUSTABLE
training_losses = []
validation_losses = []
val_labels_list = []
val_probs_list = []
training_accuracies = []
validation_accuracies = []

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).float().sum()
    return correct / labels.shape[0]


for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0
    train_correct = 0
    total_train = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(images)

        labels = labels.squeeze(1).long() # Fix for dimension mismatch, removes channel dimension and ensure it a longtensor
        loss = criterion(outputs, labels)
        loss.backward()  # backward pass
        optimizer.step()  # optimize
        train_loss += loss.item()
        # Calculate training accuracy
        acc = calculate_accuracy(outputs, labels)
        train_correct += acc * labels.size(0)
        total_train += labels.size(0)

        if batch_idx % 10 == 0:  # print every 10 batches
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}")

    train_loss /= len(train_loader)
    training_losses.append(train_loss)
    train_accuracy = train_correct / total_train
    training_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_loss = 0
        val_correct = 0
        total_val = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = labels.squeeze(1).long()  # Fix for dimension mismatch
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate necessities for roc-curve
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # For binary classification
            val_probs_list.extend(probabilities.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

            # Calculate validation accuracy
            acc = calculate_accuracy(outputs, labels)
            val_correct += acc * labels.size(0)
            total_val += labels.size(0)
        # Calculate average validation loss and accuracy for the epoch
        val_loss /= len(val_loader)
        validation_losses.append(val_loss)
        val_accuracy = val_correct / total_val
        validation_accuracies.append(val_accuracy)
        print(f"Validation Loss after Epoch {epoch+1}: {val_loss}")

# Save the model after training
torch.save(model.state_dict(), '128x128_400train_100epoch_v3.pth')
with open('training_metrics_128.pkl', 'wb') as f:
    pickle.dump({
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'training_accuracies': training_accuracies,
        'validation_accuracies': validation_accuracies,
    }, f)

print("Finished Training and saved the model, metrics")