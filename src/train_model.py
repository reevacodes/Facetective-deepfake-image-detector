import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---- STEP 1: Load Data ----
IMG_SIZE = 128
data_real = "Data/real"
data_fake = "Data/fake"

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
    return images

real_images = load_images(data_real)
fake_images = load_images(data_fake)

X = np.array(real_images + fake_images).astype("float32") / 255.0
y = np.array([0]*len(real_images) + [1]*len(fake_images))  # 0: real, 1: fake

# Class weights
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = {0: cw[0], 1: cw[1]}

# ---- STEP 2: Split & Encode ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

# ---- STEP 3: Data Augmentation ----
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

# ---- STEP 4: Model Architecture ----
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model for initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ---- STEP 5: Callbacks ----
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# ---- STEP 6: Train Initial Model ----
history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=32),
                    epochs=10,
                    validation_data=(X_test, y_test_cat),
                    class_weight=class_weights,
                    callbacks=callbacks)

# ---- STEP 7: Fine-Tune Base Model ----
base_model.trainable = True  # Unfreeze for fine-tuning
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(datagen.flow(X_train, y_train_cat, batch_size=32),
                             epochs=10,
                             validation_data=(X_test, y_test_cat),
                             class_weight=class_weights,
                             callbacks=callbacks)

# ---- STEP 8: Evaluation ----
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"[FINAL TEST] Accuracy: {acc*100:.2f}%")

# ---- STEP 9: Classification Report & Confusion Matrix ----
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=["Real", "Fake"]))

# ---- STEP 10: Save Final Model ----
model.save("deepfake_image_model.h5")

# ---- STEP 11: Accuracy Plot ----
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Val Acc')
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


