# api_gateway.py

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

IA_CONTAINER_URL = "http://ia_container:5000/process_image"


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
    app.run(host='0.0.0.0', port=8080)
