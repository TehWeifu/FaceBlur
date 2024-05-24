# ia_container.py

import io
import logging
import sys
from logging import Logger
from time import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO

IMG_SIZE = (200, 200)
PIXEL_SIZE = 5
AGE_MODEL_THRESHOLD = 0.3

app = Flask(__name__)

server_ts_start = time()
logging.basicConfig(filename='ia_container.log', level=logging.INFO)

face_detection_model = YOLO('models/FaceDetectionNet.pt')
age_model = tf.keras.models.load_model('models/AgeNet.keras')


def get_debug_logger(name: str) -> Logger:
    logging.basicConfig(level=logging.DEBUG)
    logger = Logger(name)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def detect_faces(image):
    start = time()

    results = face_detection_model(image)
    boxes = results[0].boxes

    end = time()
    logging.info(f"Face detection time: {end - start}")

    return boxes


def is_minor(image):
    start = time()

    image = image[:, :, :3]
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = age_model.predict(image, verbose=False)
    result = prediction[0][0] > AGE_MODEL_THRESHOLD

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

        # Image validation
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image = Image.open(request.files['image'])

        # Face detection
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

        # Return the image pixelated as an API response
        output = io.BytesIO()
        image = image.convert('RGB')
        image.save(output, format='JPEG')
        output.seek(0)
        return send_file(output, mimetype='image/jpeg')

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return {'error': 'Failed to process image'}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
