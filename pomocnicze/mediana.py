import os
import json
import statistics

data = 0
file_counts = []
image_extensions = {'.jpg', '.jpeg', '.png'} 

def count_image_files_in_folders(directory):
    folder_file_count = {}

    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            # Filter only image files based on extensions
            file_count = len([f for f in os.listdir(folder_path) 
                              if os.path.isfile(os.path.join(folder_path, f)) and 
                              os.path.splitext(f)[1].lower() in image_extensions])
            global data
            data += file_count
            file_counts.append(file_count)
            folder_file_count[folder] = file_count

    return folder_file_count

directory = '.././deleting_files_with_a_lot_of_samlples' 
folder_file_count = count_image_files_in_folders(directory)

if file_counts:
    median_value = statistics.median(file_counts)
    print(f"Median image file count: {median_value}")
else:
    print("No image files found.")
