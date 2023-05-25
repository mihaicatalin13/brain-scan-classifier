# Montarea Google Drive pentru a accesa setul de date
from google.colab import drive
drive.mount('/content/drive')

# Copierea set de date in Colab Workspace
!cp "/content/drive/MyDrive/brain-scan-data" "/content/" -r

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Constante si cai catre fisiere
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 1000
TRAIN_LABELS_FILE = "/content/brain-scan-data/train_labels.txt"
IMAGES_DIR = "/content/brain-scan-data/data/data"
OUTPUT_FILE = "/content/brain-scan-data/predictions.csv"

# Clasa pentru calcularea F1 Score
class F1Score(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = 0
    
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val)
        y_pred = (y_pred > 0.5).astype(int)
        f1 = f1_score(self.y_val, y_pred)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save_weights("best_model.h5")
            print(f"Epoch {epoch + 1} - F1 score improved: {f1:.4f}")
        else:
            print(f"Epoch {epoch + 1} - F1 score not improved: {f1:.4f}, Best F1 score: {self.best_f1:.4f}")

# Incarcare etichete si impartire in seturi de antrenament si validare
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

# Incarcare imagini antrenament si validare
train_images = load_images(train_labels, IMAGES_DIR)
valid_images = load_images(valid_labels, IMAGES_DIR)

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

# generator de imagini pentru augumentarea datelor
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(train_images)

# initializare callback pentru f1_score
f1_score_callback = F1Score(valid_images, valid_labels["class"])

# creare model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

callbacks = [
    f1_score_callback
]

# antrenare model 
model.fit(datagen.flow(train_images, train_labels["class"], batch_size=BATCH_SIZE),
          validation_data=(valid_images, valid_labels["class"]),
          epochs=EPOCHS,
          callbacks=callbacks)

# incarcarea celor mai bune ponderi ale modelului
model.load_weights("best_model.h5")

# predictii pe imaginile de test
test_preds = model.predict(test_images)
test_preds = (test_preds > 0.5).astype(int)

# salvare predictii intr-un fisier CSV
test_labels = pd.DataFrame({"id": test_image_ids, "class": test_preds.ravel()})
test_labels["id"] = test_labels["id"].apply(lambda x: f"{x:06d}")
test_labels.to_csv(OUTPUT_FILE, index=False)

y_pred = model.predict(valid_images)
y_pred_classes = (y_pred > 0.5).astype(int)

report = classification_report(valid_labels["class"], y_pred_classes)
conf_matrix = confusion_matrix(valid_labels["class"], y_pred_classes)

print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)