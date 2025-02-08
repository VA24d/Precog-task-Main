import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Configuration
alphabet_all = list('qwertyupasdfghjkzxcvbnmQWERTYUPKJHGFDSAZXCVBNM')
alphabet = list('qwertyupasdfghjkzxcvbnm0123456789')  # Include digits
num_alphabet = len(alphabet)
NUM_OF_LETTERS_MAX = 15  # Maximum length of CAPTCHA (can vary from 6 to 15)
IMG_ROW, IMG_COLS = 50, 135
BATCH_SIZE = 128
EPOCHS = 50
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'train')  # Path to pre-existing CAPTCHA dataset

# Dataset class
class CaptchaDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        # Extract the CAPTCHA text from the filename (assumes the label is before the first '_')
        label_text = img_name.split('_')[0]

        # Create a fixed-length label (padding with zeros if necessary)
        # Use float32 so that it matches the loss expectation
        label = np.zeros((NUM_OF_LETTERS_MAX, num_alphabet), dtype=np.float32)
        for i in range(min(len(label_text), NUM_OF_LETTERS_MAX)):
            char = label_text[i].lower()
            if char in alphabet:
                label[i, alphabet.index(char)] = 1.0

        # Read and preprocess the image
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image {img_path} not found or unable to be read.")
        img = cv2.resize(img, (IMG_COLS, IMG_ROW))

        # Reshape image to (height, width, channels)
        img = np.reshape(img, (IMG_ROW, IMG_COLS, 1)).astype('float32')

        # Apply transform if provided (e.g., ToTensor and Normalize)
        if self.transform:
            # transforms.ToTensor() expects an image with shape (H, W, C)
            img = self.transform(img)
        else:
            # If no transform is provided, manually convert to tensor and permute to channel-first format
            img = torch.from_numpy(img).permute(2, 0, 1)

        # Convert label to tensor
        label = torch.from_numpy(label)
        return img, label

# Transformation pipeline: convert to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image from (H, W, C) to (C, H, W) and scales [0,255] -> [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize the image
])

# Create Dataset and DataLoader
train_dataset = CaptchaDataset(img_dir=DATA_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define the CNN Model
class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        # After 3 layers of conv + pooling:
        # Input: (1, 50, 135)
        # After conv1 + pool: (32, 25, 67)
        # After conv2 + pool: (48, 12, 33)
        # After conv3 + pool: (64, 6, 16)
        self.fc1 = nn.Linear(64 * 6 * 16, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_alphabet * NUM_OF_LETTERS_MAX)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))  # -> (batch, 32, 25, 67)
        x = self.pool(nn.ReLU()(self.conv2(x)))  # -> (batch, 48, 12, 33)
        x = self.pool(nn.ReLU()(self.conv3(x)))  # -> (batch, 64, 6, 16)
        x = x.view(-1, 64 * 6 * 16)  # Flatten the tensor
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # Reshape to (batch, NUM_OF_LETTERS_MAX, num_alphabet)
        return x.view(-1, NUM_OF_LETTERS_MAX, num_alphabet)

# Instantiate the model
model = CaptchaCNN()

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss over each letter position in the CAPTCHA
        loss = 0.0
        for i in range(NUM_OF_LETTERS_MAX):
            loss += criterion(outputs[:, i, :], labels[:, i, :])

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'captcha_model.pth')
print('Model saved!')

# Evaluate the model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = 0.0
        for i in range(NUM_OF_LETTERS_MAX):
            loss += criterion(outputs[:, i, :], labels[:, i, :])
        test_loss += loss.item()

print(f'Evaluation Loss: {test_loss / len(train_loader):.4f}')
