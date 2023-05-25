from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/brain-scan-data" "/content/" -r

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
TRAIN_LABELS_FILE = "/content/brain-scan-data/train_labels.txt"
IMAGES_DIR = "/content/brain-scan-data/data/data"
OUTPUT_FILE = "/content/brain-scan-data/predictions.csv"

# citim etichetele de antrenare
labels = pd.read_csv(TRAIN_LABELS_FILE, index_col="id")

train_labels, valid_labels = train_test_split(labels[:17000], test_size=2000, random_state=42)

def load_images(df, img_dir):
    images = []
    for img_id in df.index:
        img_path = os.path.join(img_dir, f"{img_id:06d}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=-1)
        images.append(img)
    return np.array(images) / 255.0

# incarcam imaginile de antrenare si validare
train_images = load_images(train_labels, IMAGES_DIR)
valid_images = load_images(valid_labels, IMAGES_DIR)

# apelam reshape pentru a aduce imaginile in forma necesare pentru fit
train_images = train_images.reshape(train_images.shape[0], -1)
valid_images = valid_images.reshape(valid_images.shape[0], -1)

def load_test_images(img_dir):
    test_images = []
    test_image_ids = []
    for img_id in range(17001, 22150):
        img_path = os.path.join(img_dir, f"{img_id:06d}.png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=-1)
        test_images.append(img)
        test_image_ids.append(img_id)
    return np.array(test_images) / 255.0, test_image_ids

test_images, test_image_ids = load_test_images(IMAGES_DIR)
test_images = test_images.reshape(test_images.shape[0], -1)

# cream si antrenam modelul knn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_images, train_labels["class"])

valid_preds = knn.predict(valid_images)
f1 = f1_score(valid_labels["class"], valid_preds)
print(f"F1 score on validation set: {f1:.4f}")

# predictii pe testul de set
test_preds = knn.predict(test_images)

test_labels = pd.DataFrame({"id": test_image_ids, "class": test_preds.ravel()})
test_labels["id"] = test_labels["id"].apply(lambda x: f"{x:06d}")
test_labels.to_csv(OUTPUT_FILE, index=False)

y_pred_classes = valid_preds

report = classification_report(valid_labels["class"], y_pred_classes)
conf_matrix = confusion_matrix(valid_labels["class"], y_pred_classes)

print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)