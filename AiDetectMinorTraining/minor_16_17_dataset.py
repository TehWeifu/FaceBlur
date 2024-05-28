# Pick 25 images from a dataset of 16-17 year olds and save them to a new folder

import os
import shutil

dataset_path = './../data/face_age_16_17'

os.makedirs(dataset_path, exist_ok=True)
os.makedirs(f"{dataset_path}/adult", exist_ok=True)
os.makedirs(f"{dataset_path}/minor", exist_ok=True)

images_16 = os.listdir('./../data/face_age/016')
images_17 = os.listdir('./../data/face_age/017')

for i in range(25):
    # Copy 16 year old image
    image_16 = images_16[i]
    image_path = f'./../data/face_age/016/{image_16}'
    new_image_path = f'{dataset_path}/minor/{image_16}'
    shutil.copy(image_path, new_image_path)

    # Copy 17 year old image
    image_17 = images_17[i]
    image_path = f'./../data/face_age/017/{image_17}'
    new_image_path = f'{dataset_path}/minor/{image_17}'
    shutil.copy(image_path, new_image_path)

print('Done!')
