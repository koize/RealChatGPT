from PIL import Image
import os

def resize_images_in_place(folder, new_width):
    # Iterate over all files and subfolders in the folder
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(root, filename)
                img = Image.open(img_path)

                # Calculate the new height to maintain the aspect ratio
                width_percent = (new_width / float(img.size[0]))
                new_height = int((float(img.size[1]) * float(width_percent)))

                # Resize the image
                img = img.resize((512, 512), Image.Resampling.LANCZOS)

                # Save the resized image, overwriting the original file
                img.save(img_path)

                print(f'Resized and overwritten {filename} in {root}')

# Usage
folder = 'dataset/new'  # Main folder containing images and subfolders
new_width = 800  # Desired new width in pixels

resize_images_in_place(folder,new_width)