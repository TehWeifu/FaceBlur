import os
import shutil


def classify_images(base_folder, minor_folder, adult_folder, threshold=18):
    # Create destination folders if they do not exist
    os.makedirs(minor_folder, exist_ok=True)
    os.makedirs(adult_folder, exist_ok=True)

    # Loop through the folders named from 000 to 100
    for i in range(200):
        folder_name = f'{i:03}'  # Format the folder name with leading zeros
        current_folder = os.path.join(base_folder, folder_name)
        print(f'Processing folder {folder_name}...')

        if not os.path.isdir(current_folder):
            continue  # Skip if the folder does not exist

        # Determine the target folder based on the folder name
        target_folder = minor_folder if i < threshold else adult_folder

        # Loop through all files in the current folder
        for filename in os.listdir(current_folder):
            file_path = os.path.join(current_folder, filename)

            if os.path.isfile(file_path):
                # Move the file to the target folder
                shutil.copy(file_path, os.path.join(target_folder, filename))


if __name__ == '__main__':
    base_folder = './../data/face_age/'
    minor_folder = './../data/face_age_transformed/minor/'
    adult_folder = './../data/face_age_transformed/adult/'

    classify_images(base_folder, minor_folder, adult_folder)
