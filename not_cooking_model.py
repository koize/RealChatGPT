import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from torchsummary import summary

import cv2
import os
import numpy as np

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.img_size = 100
        self.labels_map = labels
        for label in labels:
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # Convert BGR to RGB format
                    resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size))  # Reshaping images to preferred size
                    self.data.append(resized_arr)
                    self.labels.append(class_num)
                except Exception as e:
                    print(e)
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
labels = ['apple', 'watermelon', 'banana']
train_dataset = CustomDataset(os.path.join('dataset', 'train'), labels, transform=transform)
val_dataset = CustomDataset(os.path.join('dataset', 'val'), labels, transform=transform)
test_dataset = CustomDataset(os.path.join('dataset', 'test'), labels, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.dropout(x)
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

model = CNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming your labels are numerical and continuous starting from 0
classes = np.unique(train_dataset.labels)
print (classes)
# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_dataset.labels)
print (class_weights)
#adjust class weights to penalise the over-represented class
#class_weights = [0.10, 1.42167256, 1.44033413]
#class_weights = [0.50, 1.45, 0.40]
class_weights = [0.8, 1.15, 0.9]


# Convert class weights to a tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Continue with your training as before
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 200
early_stopping_patience = 10
early_stopping_counter = 0
best_val_loss = float('inf')

train_acc = []
val_acc = []
train_loss = []
val_loss = []

model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss_item = criterion(val_outputs, val_labels)
            
            val_running_loss += val_loss_item.item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
    
    val_epoch_loss = val_running_loss / len(val_loader)
    val_epoch_acc = val_correct / val_total
    val_loss.append(val_epoch_loss)
    val_acc.append(val_epoch_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
    
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(), os.path.join('model', 'real_chatgpt.pth'))
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping")
        break

# Plotting
epochs_range = range(len(train_acc))

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.eval()

# Initialize variables to track test performance
test_loss = 0.0
test_correct = 0
test_total = 0

# No gradient needed for evaluation
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        
        # Forward pass
        test_outputs = model(test_images)
        loss = criterion(test_outputs, test_labels)
        
        # Update test loss
        test_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(test_outputs.data, 1)
        test_total += test_labels.size(0)
        test_correct += (predicted == test_labels).sum().item()

# Calculate average loss and accuracy
test_loss /= len(test_loader)
test_accuracy = test_correct / test_total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Optional: Generate a confusion matrix
all_preds = []
all_targets = []
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        outputs = model(test_images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(test_labels.cpu().numpy())

conf_matrix = confusion_matrix(all_targets, all_preds)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
# Display the summary
#  (3, 100, 100) refers to the input size, with the input images being of 100x100 with 3 channels (RGB).
summary(model,(3,100,100))