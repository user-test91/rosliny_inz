import zipfile

zip_file_path = '.././cleanSet.zip'

extract_to_path = './'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f"Files extracted to '{extract_to_path}'")
