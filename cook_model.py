import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


from sklearn.metrics import classification_report,confusion_matrix


import cv2
import os

import numpy as np

labels = ['apple', 'watermelon']
img_size = 100
def get_data(data_dir):
    data = []
    labels_list = []
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1]  # Convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append(resized_arr)
                labels_list.append(class_num)
            except Exception as e:
                print(e)
    return np.array(data), np.array(labels_list)

# Usage
train_images, train_labels = get_data('D:/SP/mlai/projek/PROPOGANDA/dataset/train/')
val_images, val_labels = get_data('D:/SP/mlai/projek/PROPOGANDA/dataset/train/')  # Should this be 'dataset/val/'?
test_images, test_labels = get_data('D:/SP/mlai/projek/PROPOGANDA/dataset/test/')

l = []
for label in train_labels:
    if label == 0:
        l.append("apple")
    else:
        l.append("watermelon")
sns.set_style('darkgrid')
sns.countplot(l)

plt.figure(figsize=(5,5))
plt.imshow(train_images[1])  # Access the second image
plt.title(labels[train_labels[1]])  # Use the label at the same index to get the label name

plt.figure(figsize=(5,5))
plt.imshow(train_images[-1])  # Access the last image
plt.title(labels[train_labels[-1]])  # Use the label at the same index to get the label name

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in zip(train_images, train_labels):
  x_train.append(feature)
  y_train.append(label)

for feature, label in zip(val_images, val_labels):
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(100,100,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = Adam(learning_rate=0.0001)  # Adjusted learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
tf.keras.models.save_model(model, 'D:/SP/mlai/projek/PROPOGANDA/model/real_chatgpt.h5')
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(enumerate(class_weights))
class_weights_dict = {0: 1.05, 1: 1.0}  # Manually set class weights to 1.0 for both classes
history = model.fit(x_train, y_train, epochs=500, class_weight=class_weights_dict, validation_data=(x_val, y_val), callbacks=[early_stopping])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))  # This adjusts the range to the actual number of epochs trained

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict(x_val)
predictions = predictions.reshape(1,-1)[0]
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_labels
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d')
print(classification_report(y_val, predictions, target_names = ['Apple (Class 0)','Watermelon (Class 1)']))