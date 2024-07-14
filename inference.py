from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import os
import sys
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix
from torchsummary import summary

import cv2
import os
import numpy as np

label = ''
frame = None

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

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def import_and_predict(image_data, model):
    size = (100, 100)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)

    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Change HWC to CHW format and add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        prediction = model(image)
        prediction = F.softmax(prediction, dim=1)  # Apply softmax to get probabilities
    
    return prediction.cpu().numpy()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel()
model.load_state_dict(torch.load(os.path.join('model', 'real_chatgpt.pth')))
model = model.to(device)  # Move model to the correct device
model.eval()

# Initialize variables to track test performance
test_loss = 0.0
test_correct = 0
test_total = 0
class_weights = [0.10, 1.65607345, 0.96980976]
# Convert class weights to a tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cap = cv2.VideoCapture(1)

if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

class_labels = ["apple", "watermelon", "banana"]

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (100, 100))
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')

    prediction = import_and_predict(image, model)

    confidence_threshold = 0.7  # Adjusted based on model evaluation

    max_confidence_score = np.max(prediction)
    predicted_class_idx = np.argmax(prediction)

    # Check if the highest confidence score is below the threshold
    if max_confidence_score < confidence_threshold:
        predict = "unknown"
        confidence_text = "unknown"
    else:
        predict = class_labels[predicted_class_idx]
        confidence_text = f"{max_confidence_score*100:.2f}%"

    # Display all class confidences
    y_position = 30
    for i, (label, score) in enumerate(zip(class_labels, prediction[0])):
        text = f"{label}: {score*100:.2f}%"
        cv2.putText(original, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_position += 30

    cv2.putText(original, f"Prediction: {predict}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    y_position += 30
    cv2.putText(original, f"Confidence: {confidence_text}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()