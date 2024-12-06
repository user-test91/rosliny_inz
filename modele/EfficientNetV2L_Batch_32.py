import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

tf.keras.mixed_precision.set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()

train_dir = '.././train'
val_dir = '.././valid'

seed = 22
batch_size_per_replica = 16
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
epochs = 100

def load_and_preprocess(image, label, training=False):
    image = preprocess_input(image)
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image, label

train_dataset_raw = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(480, 480),
    batch_size=None,
    seed=seed,
    label_mode='int'
)

class_names = train_dataset_raw.class_names

train_dataset = train_dataset_raw.map(lambda x, y: load_and_preprocess(x, y, training=True), num_parallel_calls=tf.data.AUTOTUNE)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(480, 480),
    batch_size=None,
    seed=seed,
    label_mode='int'
).map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)


train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
with strategy.scope():
    base_model = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(480, 480, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(class_names), activation='softmax', dtype='float32')(x)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=1000, alpha=0.01
    )
    
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_folder = f'EfficientNetV2L_Batch_{batch_size}_V2'
os.makedirs(batch_folder, exist_ok=True)

checkpoint_path = os.path.join(batch_folder, 'model_checkpoint.h5')
final_model_path = os.path.join(batch_folder, 'final_trained_model.h5')
log_path = os.path.join(batch_folder, 'training_log.csv')
history_plot_path = os.path.join(batch_folder, 'training_history.png')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
csv_logger = CSVLogger(log_path, append=False)

start_time = datetime.datetime.now()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint, csv_logger]
)

end_time = datetime.datetime.now()
training_duration = (end_time - start_time).total_seconds() / 60

model.save(final_model_path)

training_log = pd.read_csv(log_path)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(training_log['epoch'], training_log['accuracy'], label='Training Accuracy')
plt.plot(training_log['epoch'], training_log['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_log['epoch'], training_log['loss'], label='Training Loss')
plt.plot(training_log['epoch'], training_log['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(history_plot_path)
plt.close()

print(f"Training completed in {training_duration:.2f} minutes.")
print(f"Final model saved to {final_model_path}")
