import os
import random
import shutil

# Define the source and destination directories
source_dir = './../data/face_age_transformed'
destination_dir = './../data/face_age_reduced'

# Categories (folders) in the dataset
categories = ['minor', 'adult']

# Number of pictures to copy from each category
num_pictures = 50

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for category in categories:
    # Define the full paths for source and destination category folders
    src_category_path = os.path.join(source_dir, category)
    dest_category_path = os.path.join(destination_dir, category)

    # Create the category folder in the destination directory if it doesn't exist
    if not os.path.exists(dest_category_path):
        os.makedirs(dest_category_path)

    # Get a list of all files in the source category folder
    all_files = os.listdir(src_category_path)

    # Select a random sample of files from the list
    selected_files = random.sample(all_files, num_pictures)

    # Copy each selected file to the destination category folder
    for file_name in selected_files:
        src_file_path = os.path.join(src_category_path, file_name)
        dest_file_path = os.path.join(dest_category_path, file_name)
        shutil.copy(src_file_path, dest_file_path)

print(f"Dataset replication complete with {num_pictures} pictures per category.")
