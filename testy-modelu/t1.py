import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import csv
import os

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

model = tf.keras.models.load_model('../model/EfficientNetV2L_Batch_32_V2/final_trained_model.h5')

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '.././test',
    labels='inferred',
    label_mode='int',
    batch_size=1,
    image_size=(480, 480),
    shuffle=False
)

class_names = test_dataset.class_names
num_classes = len(class_names)

per_class_correct_top1 = np.zeros(num_classes)
per_class_correct_top5 = np.zeros(num_classes)
per_class_total = np.zeros(num_classes)

all_labels = []
all_preds_top1 = []

for images, labels in test_dataset:
    images = images.numpy()
    label = labels.numpy()[0]
    per_class_total[label] += 1

    outputs_sum = np.zeros(num_classes)

    for _ in range(5):
        augmented_image = data_augmentation(images)
        outputs = model.predict(augmented_image)
        outputs = outputs[0]
        outputs_sum += outputs

    outputs_avg = outputs_sum / 5
    top5_preds = np.argsort(outputs_avg)[-5:][::-1]
    top1_pred = top5_preds[0]

    if top1_pred == label:
        per_class_correct_top1[label] += 1

    if label in top5_preds:
        per_class_correct_top5[label] += 1

    all_labels.append(label)
    all_preds_top1.append(top1_pred)

per_class_accuracy_top1 = per_class_correct_top1 / per_class_total
per_class_accuracy_top5 = per_class_correct_top5 / per_class_total

overall_accuracy_top1 = np.sum(per_class_correct_top1) / np.sum(per_class_total)
overall_accuracy_top5 = np.sum(per_class_correct_top5) / np.sum(per_class_total)

os.makedirs('top1_results', exist_ok=True)
os.makedirs('top5_results', exist_ok=True)

with open('top1_results/class_statistics.txt', 'w') as f:
    for i, class_name in enumerate(class_names):
        f.write(f'Class: {class_name}\n')
        f.write(f'Number of samples in class: {int(per_class_total[i])}\n')
        f.write(f'Number of correctly predicted samples: {int(per_class_correct_top1[i])}\n')
        f.write(f'Per-class accuracy: {per_class_accuracy_top1[i]*100:.2f}%\n\n')

with open('top1_results/overall_statistics.txt', 'w') as f:
    f.write(f'Total samples: {int(np.sum(per_class_total))}\n')
    f.write(f'Total predictions: {int(np.sum(per_class_total))}\n')
    f.write("All samples have been predicted.\n")
    f.write(f'Overall accuracy: {overall_accuracy_top1*100:.2f}%\n')

cm_top1 = confusion_matrix(all_labels, all_preds_top1)
save_confusion_matrix(cm_top1, class_names, 'top1_results/confusion_matrix.png', title='Confusion Matrix (Top-1)')

with open('top1_results/per_class_accuracy.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['class_name', 'accuracy'])
    for i, class_name in enumerate(class_names):
        writer.writerow([class_name, f"{per_class_accuracy_top1[i]*100:.2f}%"])

with open('top5_results/class_statistics.txt', 'w') as f:
    for i, class_name in enumerate(class_names):
        f.write(f'Class: {class_name}\n')
        f.write(f'Number of samples in class: {int(per_class_total[i])}\n')
        f.write(f'Number of correctly predicted samples: {int(per_class_correct_top5[i])}\n')
        f.write(f'Per-class accuracy: {per_class_accuracy_top5[i]*100:.2f}%\n\n')

with open('top5_results/overall_statistics.txt', 'w') as f:
    f.write(f'Total samples: {int(np.sum(per_class_total))}\n')
    f.write(f'Total predictions: {int(np.sum(per_class_total))}\n')
    f.write("All samples have been predicted.\n")
    f.write(f'Overall accuracy: {overall_accuracy_top5*100:.2f}%\n')

with open('top5_results/per_class_accuracy.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['class_name', 'accuracy'])
    for i, class_name in enumerate(class_names):
        writer.writerow([class_name, f"{per_class_accuracy_top5[i]*100:.2f}%"])
