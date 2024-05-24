# ia_container.py

import io
import logging
from time import time

import numpy as np
from PIL import Image
from flask import Flask, request, send_file
from ultralytics import YOLO

PIXEL_SIZE = 5

app = Flask(__name__)

server_ts_start = time()
logging.basicConfig(filename='ia_container.log', level=logging.INFO)

face_detection_model = YOLO('FaceDetection/models/face-detection.pt')


def detect_faces(image):
    start = time()

    results = face_detection_model(image)
    boxes = results[0].boxes

    end = time()
    logging.info(f"Face detection time: {end - start}")

    return boxes


def is_minor(image):
    start = time()

    result = True

    end = time()
    logging.info(f"Minor prediction time: {end - start}")

    return result


def pixel_face(face_image):
    face_image = Image.fromarray(face_image)

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

    pixelated_image = np.array(pixelated_image)

    return pixelated_image


@app.route('/hc', methods=['GET'])
def hc():
    server_ts_current = time()
    return {'status': 'ok', 'uptime': server_ts_current - server_ts_start}


@app.route('/blur', methods=['POST'])
def blur():
    try:
        request_ts_start = time()

        image = Image.open(request.files['image'])

        boxes = detect_faces(image)

        for box in boxes:
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])

            # Create a new image with the face detected
            img_array = np.array(image)
            face = img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            if is_minor(face):
                face = pixel_face(face)

            img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = face
            image = Image.fromarray(img_array)

        image = image.convert('RGB')
        output = io.BytesIO()
        image.save(output, format='JPEG')
        output.seek(0)
        return send_file(output, mimetype='image/jpeg')
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return {'error': 'Failed to process image'}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
