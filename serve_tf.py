from PIL import Image, ImageOps
import tensorflow as tf

import cv2
import numpy as np
import os
import sys



label = ''

frame = None

def import_and_predict(image_data, model):
    
        size = (100,100)    
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('D:/SP/mlai/projek/PROPOGANDA/model/real_chatgpt.h5')

    
cap = cv2.VideoCapture(1)

if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (100, 100))
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')

    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    prediction = import_and_predict(image, model)
    #print(prediction)

    confidence_threshold = 0.1  # Adjusted based on model evaluation

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
        predict = "real unknown"  # Fallback case, though it should not be reached

    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(original, f"{max_confidence_score*100:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()