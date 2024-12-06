import numpy as np
import os
import cv2
import albumentations as A

transform = A.Compose([
    A.Flip(p=0.5),
    A.Rotate(limit=15, p=0.5),  
    A.RandomBrightnessContrast(p=0.5), 
    A.HueSaturationValue(p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.5)
    ], p=0.3), 
])

target_images = 450

production_images_folder_path = "./cleanSet"  

class_folders = [folder for folder in os.listdir(production_images_folder_path)
                 if os.path.isdir(os.path.join(production_images_folder_path, folder))]

for class_name in class_folders:
    path_to_img_folder = os.path.join(production_images_folder_path, class_name)
    image_files = [image for image in os.listdir(path_to_img_folder)
                   if os.path.isfile(os.path.join(path_to_img_folder, image))]

    num_existing_images = len(image_files)
    if num_existing_images >= target_images:
        print(f"Folder '{class_name}' has {num_existing_images} images. No augmentation needed.")
        continue

    num_augmentations_needed = target_images - num_existing_images
    print(f"Folder '{class_name}' has {num_existing_images} images. Generating {num_augmentations_needed} augmented images...")

    for _ in range(num_augmentations_needed):
        image_file = np.random.choice(image_files)
        image_path = os.path.join(path_to_img_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image {image_path}. Skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = transform(image=image)
        augmented_image = augmented['image']
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

        augmented_image_filename = f"aug_{np.random.randint(1e6)}_{image_file}"
        augmented_image_path = os.path.join(path_to_img_folder, augmented_image_filename)
        cv2.imwrite(augmented_image_path, augmented_image)

print("Augmentation complete.")
