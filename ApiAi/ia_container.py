# ia_container.py: API that receives an image and returns the same image with the minor faces pixelated

import io
import logging
import sys
import uuid
from datetime import datetime
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

face_detection_model = YOLO('models/FaceDetectionNet.pt')
age_model = tf.keras.models.load_model('models/AgeNet.keras')


def get_debug_logger(name: str) -> Logger:
    logging.basicConfig(level=logging.DEBUG)
    logger = Logger(name)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def get_formated_mili_seconds(time_stamp: float) -> str:
    ts_ms = int(time_stamp * 1000)
    return f"{ts_ms} ms"


def validate_request_image():
    if 'image' not in request.files:
        return False, jsonify({"error": "No image file provided"})

    image_file = request.files['image']

    # FIXME: Mime type does not work when image if forwarded the api gateway
    # if image_file.mimetype not in ['image/jpeg', 'image/png']:
    #     return False, jsonify({"error": "Unsupported image type"})

    try:
        image = Image.open(image_file)
        image.verify()
        return True, None
    except Exception as e:
        return False, jsonify({"error": "Invalid image file"})


def detect_faces(image, logger: Logger):
    start = time()

    results = face_detection_model(image)
    boxes = results[0].boxes

    end = time()
    logger.info(f"Face detection found {len(boxes)} faces in {get_formated_mili_seconds(end - start)}")

    return boxes


def predict_minor_score(image):
    image = image[:, :, :3]
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    start = time()
    prediction = age_model.predict(image, verbose=False)
    end = time()

    return prediction[0][0], end - start


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
    request_ts_start = time()
    request_uuid = f"request_{uuid.uuid4()}"
    request_logger = get_debug_logger(request_uuid)
    request_logger.info(f"Processing {request_uuid} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Validate the request image and load it
        is_image_valid, response = validate_request_image()
        if not is_image_valid:
            return response, 400
        image = Image.open(request.files['image'])

        # Faces detection
        boxes = detect_faces(image, request_logger)

        for idx, box in enumerate(boxes):
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])

            # Create a new image with the face detected
            img_array = np.array(image)
            face = img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            minor_score, elapsed_time = predict_minor_score(face)
            is_minor = minor_score > AGE_MODEL_THRESHOLD
            if is_minor:
                face = pixel_face(face)

            request_logger.info(
                f"Face {idx + 1} detected at ({top_left_x}, {top_left_y}) - ({bottom_right_x}, {bottom_right_y})")
            request_logger.info(
                f"Face {idx + 1} predicted as {'minor' if is_minor else 'adult'} (score: {minor_score:.2f}) in {get_formated_mili_seconds(elapsed_time)}")

            img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = face
            image = Image.fromarray(img_array)

        # Prepare the return image pixelated as an API response
        output = io.BytesIO()
        image = image.convert('RGB')
        image.save(output, format='JPEG')
        output.seek(0)
        response = send_file(output, mimetype='image/jpeg')

        request_ts_end = time()
        request_logger.info(f"Request processing time: {request_ts_end - request_ts_start}")

        return response

    except Exception as e:
        request_logger.error(f"Error processing image for request {request_uuid}: {e}")
        return {'error': 'Failed to process image'}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
