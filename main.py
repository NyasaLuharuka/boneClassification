import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential

batch_size = 32
image_height = 180
image_width = 180
dataset = 'boneBreakData'

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (image_height, image_width),
    batch_size = batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="validation",
    image_size = (image_height, image_width),
    batch_size=batch_size,
    seed=123
)

class_names = train_ds.class_names

model = Sequential([
    layers.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save('boneClassification.h5')