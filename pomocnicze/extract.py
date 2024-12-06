import os
import shutil
import random

num_samples = 50
source_directory = './cutedDown_100'
test_directory = './test'

os.makedirs(test_directory, exist_ok=True)

for class_name in os.listdir(source_directory):
    class_source_path = os.path.join(source_directory, class_name)
    
    
    if os.path.isdir(class_source_path):
        class_test_path = os.path.join(test_directory, class_name)
        os.makedirs(class_test_path, exist_ok=True)
      
        files = os.listdir(class_source_path)

        files = [f for f in files if os.path.isfile(os.path.join(class_source_path, f))]

        random.shuffle(files)

        num_files_to_move = min(num_samples, len(files))

        files_to_move = files[:num_files_to_move]

        for filename in files_to_move:
            src_file = os.path.join(class_source_path, filename)
            dst_file = os.path.join(class_test_path, filename)
            shutil.move(src_file, dst_file)
        
        print(f"Moved {num_files_to_move} samples from class '{class_name}' to the test directory.")
