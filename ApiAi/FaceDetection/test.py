import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

PIXEL_SIZE = 5

img_path = 'teens.jpg'
img = cv2.imread(img_path)

model = YOLO('models/face-detection.pt')
results = model(img_path)
boxes = results[0].boxes

for box in boxes:
    top_left_x = int(box.xyxy.tolist()[0][0])
    top_left_y = int(box.xyxy.tolist()[0][1])
    bottom_right_x = int(box.xyxy.tolist()[0][2])
    bottom_right_y = int(box.xyxy.tolist()[0][3])

    # Create a new image with the face detected
    face = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    face_image = Image.fromarray(face)

    width, height = face_image.size

    # Resize the image to a smaller size
    small_face = face_image.resize(
        (width // PIXEL_SIZE, height // PIXEL_SIZE),
        resample=Image.NEAREST
    )

    # Resize the image back to its original size
    pixelated_image = small_face.resize(
        (width, height),
        resample=Image.NEAREST
    )

    # Change color so it doesnt appear blue
    pixelated_image = np.array(pixelated_image)

    # Replace the face with the pixelated image
    img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = pixelated_image

cv2.imwrite('pixelated_image.png', img)
