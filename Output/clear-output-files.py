import os

# Specify the folder path
folder_path = "/home/2023/mnadea52/Winter2025/Deep-Reinforcement-Learning/Output"

# Loop through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.out') or filename.endswith('.err'):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"Deleted {file_path}")
        
# Loop through the folders in the folder
for folder in os.listdir(folder_path):
    folder_path_to_check = os.path.join(folder_path, folder)
    if os.path.isdir(folder_path_to_check) and folder == 'output':
        for root, dirs, files in os.walk(folder_path_to_check, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path_to_check)
        print(f"Deleted {folder_path_to_check} and all of its content")