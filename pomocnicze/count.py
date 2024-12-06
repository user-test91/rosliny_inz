import os
import json
import statistics

data = 0
file_counts = []

def count_files_in_folders(directory):
    folder_file_count = {}

    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            global data
            data += file_count 
            file_counts.append(file_count) 
            folder_file_count[folder] = file_count

    return folder_file_count

directory = './dataset' 
folder_file_count = count_files_in_folders(directory)
print(data)
if file_counts:
    median_value = statistics.median(file_counts)
    print(f"Median file count: {median_value}")

