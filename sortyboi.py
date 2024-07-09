import os
import shutil
import random

def split_photos(input_folder, output_folders, split_ratio):
    # Create output directories if they don't exist
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)
    
    # Get a list of all files in the input folder
    all_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Shuffle the files randomly
    random.shuffle(all_files)
    
    # Calculate the number of files for each split
    total_files = len(all_files)
    train_count = int(total_files * split_ratio[0])
    test_count = int(total_files * split_ratio[1])
    val_count = total_files - train_count - test_count
    
    # Split the files into the respective folders
    train_files = all_files[:train_count]
    test_files = all_files[train_count:train_count + test_count]
    val_files = all_files[train_count + test_count:]
    
    # Copy the files to the respective folders
    for f in train_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(output_folders[0], f))
    for f in test_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(output_folders[1], f))
    for f in val_files:
        shutil.copy(os.path.join(input_folder, f), os.path.join(output_folders[2], f))

# Define the input and output folders and the split ratio
input_folder = r'D:/Repo/shared_repo/RealChatGPT/sorting'
output_folders = [
    r'D:/Repo/shared_repo/RealChatGPT/dataset/test/apple',
    r'D:/Repo/shared_repo/RealChatGPT/dataset/train/apple',
    r'D:/Repo/shared_repo/RealChatGPT/dataset/val/apple'
]
split_ratio = [0.70, 0.15, 0.15]

# Call the function to split the photos
split_photos(input_folder, output_folders, split_ratio)