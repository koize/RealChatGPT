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
    val_count = int(total_files * split_ratio[1])
    test_count = total_files - train_count - val_count
    
    # Split the files into the respective folders
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]
    
    # Copy the files to the respective folders if they don't already exist there
    for f in train_files:
        if not os.path.exists(os.path.join(output_folders[0], f)):
            shutil.copy(os.path.join(input_folder, f), os.path.join(output_folders[0], f))
    for f in val_files:
        if not os.path.exists(os.path.join(output_folders[1], f)):
            shutil.copy(os.path.join(input_folder, f), os.path.join(output_folders[1], f))
    for f in test_files:
        if not os.path.exists(os.path.join(output_folders[2], f)):
            shutil.copy(os.path.join(input_folder, f), os.path.join(output_folders[2], f))

# Define the input and output folders and the split ratio
input_base_folder = 'sorting'
output_base_folder = 'dataset'

fruits = ['apple', 'watermelon', 'banana']
split_ratio = [0.70, 0.15, 0.15]

# Process each fruit folder
for fruit in fruits:
    input_folder = os.path.join(input_base_folder, fruit)
    output_folders = [
        os.path.join(output_base_folder, 'train', fruit),
        os.path.join(output_base_folder, 'val', fruit),
        os.path.join(output_base_folder, 'test', fruit)
    ]
    split_photos(input_folder, output_folders, split_ratio)