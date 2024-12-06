import os

def fill_directory_names(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    max_number = max(int(d) for d in directories)
    num_digits = len(str(max_number))

    for directory in directories:
        new_name = directory.zfill(num_digits)
        os.rename(os.path.join(path, directory), os.path.join(path, new_name))
        print(f"Renamed {directory} to {new_name}")

dataset_path = '.././working_images'
fill_directory_names(dataset_path)