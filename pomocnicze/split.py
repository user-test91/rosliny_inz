import os
import shutil
import random

dataset_dir = './cleanSet'
train_dir = './train'
val_dir = './valid'

random.seed(22)

train_images_per_class = 350
val_images_per_class = 100

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    images = [f for f in os.listdir(class_dir)
              if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(class_dir, f))]
    random.shuffle(images) 

    num_images = len(images)
    if num_images < train_images_per_class + val_images_per_class:
        print(f"Class '{class_name}' does not have enough images. Required: {train_images_per_class + val_images_per_class}, Found: {num_images}")
        continue
    train_images = images[:train_images_per_class]
    val_images = images[train_images_per_class:train_images_per_class + val_images_per_class]

    for image in train_images:
        src = os.path.join(class_dir, image)
        dst = os.path.join(train_class_dir, image)
        shutil.move(src, dst)

    for image in val_images:
        src = os.path.join(class_dir, image)
        dst = os.path.join(val_class_dir, image)
        shutil.move(src, dst)

    print(f"Processed class '{class_name}': {len(train_images)} training images, {len(val_images)} validation images.")

    remaining_images = images[train_images_per_class + val_images_per_class:]
    if remaining_images:
        print(f"Class '{class_name}' has {len(remaining_images)} remaining images not used.")

print("Dataset splitting complete.")
