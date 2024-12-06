import json
import os
import matplotlib.pyplot as plt

directory = ".././cut down 121 classes 410 pic max - clean"
image_extensions = {'.jpg', '.jpeg', '.png'}
folder_file_count = {}

total_file_count = 0
for root, dirs, files in os.walk(directory):
    print(dirs)
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        file_count = len([f for f in os.listdir(folder_path) 
                          if os.path.isfile(os.path.join(folder_path, f)) and 
                          os.path.splitext(f)[1].lower() in image_extensions])
        total_file_count += file_count
        folder_file_count[folder] = file_count

min_class = min(folder_file_count, key=folder_file_count.get)
min_count = folder_file_count[min_class]
print(f"Klasa z najmniejszą liczbą próbek: {min_class}, liczba próbek: {min_count}")

with open('aug.json', 'w') as json_file:
    json.dump(folder_file_count, json_file, indent=4)

folders = list(folder_file_count.keys())
file_counts = list(folder_file_count.values())

plt.figure(figsize=(12, 6))
plt.bar(folders, file_counts, color='blue')
plt.xlabel('Folders (Classes)')
plt.ylabel('Number of Pictures')
plt.title(f'Total amount of samples: {total_file_count}')

plt.xticks(rotation=45, ha='right', fontsize=8)

plt.savefig('clean.pdf')
plt.show()
