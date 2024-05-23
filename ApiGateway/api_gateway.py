# api_gateway.py

import time

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

IA_CONTAINER_URL = "http://ia_container:5001/process_image"

time_start = time.time()


@app.route('/hc', methods=['GET'])
def hc():
    time_current = time.time()
    return jsonify({"status": "ok", "uptime": time_current - time_start})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    response = requests.post(IA_CONTAINER_URL, files={'image': image})

    if response.status_code == 200:
        return response.content, 200, {'Content-Type': 'image/jpeg'}
    else:
        return jsonify({"error": "Failed to process image"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
