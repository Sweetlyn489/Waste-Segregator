# train_biodeg_classifier.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import os

# === SETTINGS ===
DATA_DIR = "Dataset"            # folder with train/val/test subfolders
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-4
MODEL_OUT = "biodeg_mobilenetv2.h5"

# === Data generators with augmentation ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=18,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.12,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"   # 2 classes
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# === Build model ===
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False  # freeze base initially

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")  # binary
])

model.compile(optimizer=Adam(learning_rate=LR),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

model.summary()

# === Callbacks ===
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT, save_best_only=True, monitor="val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True)
]

# === Train ===
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# Optionally unfreeze some base layers and fine-tune
base.trainable = True
# fine-tune from this layer onwards (tune empirically)
fine_tune_at = 100
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LR/10),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

history_fine = model.fit(
    train_gen,
    epochs=EPOCHS//2,
    validation_data=val_gen,
    callbacks=callbacks
)

# Save final model (already saved by checkpoint, but save again)
model.save("biodeg_final.h5")
print("Saved model to biodeg_final.h5")
