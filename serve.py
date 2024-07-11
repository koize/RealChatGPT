from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import os
import sys

label = ''
frame = None

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
model = CNNModel()
model.load_state_dict(torch.load(os.path.join('model', 'real_chatgpt.pth')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cap = cv2.VideoCapture(0)

if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

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
    elif predicted_class_idx == 0:
        predict = "apple!"
    elif predicted_class_idx == 1:
        predict = "watermelon!"
    elif predicted_class_idx == 2:
        predict = "banana!"
    else:
        predict = "really unknown"  # Fallback case, though it should not be reached

    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(original, f"{max_confidence_score*100:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()