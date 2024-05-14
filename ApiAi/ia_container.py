# ia_container.py
from flask import Flask, request, send_file
from PIL import Image, ImageDraw
import io
import time
import logging

app = Flask(__name__)

logging.basicConfig(filename='ia_container.log', level=logging.INFO)


def detect_faces(image):
    # Simulación de detección de caras (coordenadas aleatorias)
    return [(50, 50, 150, 150), (200, 200, 300, 300)]


def is_minor(face_image):
    # Simulación de predicción de menor de edad
    return True


@app.route('/process_image', methods=['POST'])
def process_image():
    start_time = time.time()
    image = Image.open(request.files['image'])
    faces = detect_faces(image)
    face_detection_time = time.time() - start_time

    draw = ImageDraw.Draw(image)
    log_data = {
        'request_time': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        'faces': [],
        'face_detection_time_ms': face_detection_time * 1000,
        'age_prediction_time_ms': 0,
        'total_time_ms': 0
    }

    age_prediction_start_time = time.time()
    for face in faces:
        face_image = image.crop(face)
        if is_minor(face_image):
            draw.rectangle(face, fill='black')  # Pixelar cara
        log_data['faces'].append({
            'coordinates': face,
            'is_minor': is_minor(face_image)
        })

    age_prediction_time = time.time() - age_prediction_start_time
    total_time = time.time() - start_time

    log_data['age_prediction_time_ms'] = age_prediction_time * 1000
    log_data['total_time_ms'] = total_time * 1000

    logging.info(log_data)

    output = io.BytesIO()
    image.save(output, format='JPEG')
    output.seek(0)
    return send_file(output, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
