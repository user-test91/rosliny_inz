import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import json
from tensorflow.keras.regularizers import l2
tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "physical gpu,", len(logical_gpus), "logical gpus")

batch_sizes = [32]
epochs = 40

preprocess_input = resnet50.preprocess_input

def save_confusion_matrix(cm, class_names, file_path, title='Confusion Matrix', cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(15, 15)) 
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_title(title)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    
    ax.set_xticklabels(class_names, rotation=90, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] != 0:
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.savefig(file_path, bbox_inches='tight')  
    plt.close() 

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

main_folder = '../dataset'
train_path = os.path.join(main_folder, 'train')
test_path = os.path.join(main_folder, 'test')
valid_path = os.path.join(main_folder, 'valid')

batch_size = 32
img_size = (224, 224)
epochs = 100
preprocess_input = resnet50.preprocess_input

train_dataset = image_dataset_from_directory(
    train_path, image_size=img_size, batch_size=batch_size
)
class_names = train_dataset.class_names

train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)

valid_dataset = image_dataset_from_directory(
    valid_path, image_size=img_size, batch_size=batch_size
).map(lambda x, y: (preprocess_input(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = image_dataset_from_directory(
    test_path, image_size=img_size, batch_size=batch_size
).map(lambda x, y: (preprocess_input(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)

for batch_size in batch_sizes:
    print(f"\n starting training with batch size: {batch_size}")

    batch_folder = f'CNN_Batch_{batch_size}_teraz82'
    print(batch_folder)
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)

    checkpoint_path = os.path.join(batch_folder, 'model_checkpoint.h5')
    log_path = os.path.join(batch_folder, 'training_log.csv')
    history_plot_path = os.path.join(batch_folder, 'training_history.png')
    stats_path = os.path.join(batch_folder, 'training_stats.json')
    tensorboard_log_dir = os.path.join(batch_folder, 'tensorboard_logs')
    training_successful = False
    while not training_successful:
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print("restart training.")

            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base_model.trainable = True
            fine_tune_at = 144
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
            predictions = Dense(len(class_names), activation='softmax')(x)

            model = Model(inputs=base_model.input, outputs=predictions)
            fine_tune_lr = 1e-5
            optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() 
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

            
            model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
            csv_logger = CSVLogger(log_path, append=False)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, update_freq='epoch', histogram_freq=5)

            start_time = datetime.datetime.now()

            history = model.fit(
                train_dataset,
                validation_data=valid_dataset,
                epochs=epochs,
                callbacks=[early_stopping, model_checkpoint, csv_logger],
                initial_epoch=0
            )

            end_time = datetime.datetime.now()
            training_duration = (end_time - start_time).total_seconds() / 60

            final_model_path = os.path.join(batch_folder, f'resnet_model_batch_{batch_size}.h5')
            model.save(final_model_path)
            print(f"final model saved to {final_model_path}")

            import pandas as pd
            training_log = pd.read_csv(log_path)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(training_log['epoch'], training_log['accuracy'], label='training accuracy')
            plt.plot(training_log['epoch'], training_log['val_accuracy'], label='validation accuracy')
            plt.title('model accuracy')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(training_log['epoch'], training_log['loss'], label='training loss')
            plt.plot(training_log['epoch'], training_log['val_loss'], label='validation loss')
            plt.title('model loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()

            plt.savefig(history_plot_path)
            plt.close()
            print(f"training history plot saved to {history_plot_path}")

            best_epoch = training_log['val_loss'].idxmin()
            best_val_loss = training_log.loc[best_epoch, 'val_loss']
            best_val_accuracy = training_log.loc[best_epoch, 'val_accuracy']

            training_stats = {
                'batch_size': batch_size,
                'training_duration_minutes': training_duration,
                'best_epoch': int(best_epoch),
                'best_val_loss': float(best_val_loss),
                'best_val_accuracy': float(best_val_accuracy)
            }

            with open(stats_path, 'w') as f:
                json.dump(training_stats, f, indent=4)
            print(f"statistics saved to {stats_path}")

            print(f"training with batch size {batch_size} completed.\n")

            print(f"evaluating model on test data with batch size {batch_size}...")
            y_true = np.concatenate([y for x, y in test_dataset], axis=0)
            y_pred = np.argmax(model.predict(test_dataset), axis=-1)

            conf_matrix = confusion_matrix(y_true, y_pred)

            file_path = os.path.join(batch_folder, f'confusion_matrix_batch_{batch_size}.png')
            save_confusion_matrix(conf_matrix, class_names, file_path)
            print(f"confusion matrix saved to {file_path}")

            training_successful = True
            tf.keras.backend.clear_session()
            del model

        except Exception as e:
            print(f"an error occurred with batch size {batch_size}: {e}")
            print("restarting training for batch size...")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

print("all trainings completed.")

