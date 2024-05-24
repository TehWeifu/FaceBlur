# api_gateway.py

import logging
from time import time

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI model URL
AI_CONTAINER_URL = "http://ai_model:5001"  # Docker URL
# AI_CONTAINER_URL = "http://localhost:5001"  # Local URL

ts_server_start = time()


@app.route('/hc', methods=['GET'])
def hc():
    time_current = time()
    return jsonify({"status": "ok", "uptime": time_current - ts_server_start})


@app.route('/hc_model', methods=['GET'])
def hc_model():
    try:
        response = requests.get(f"{AI_CONTAINER_URL}/hc")
        response.raise_for_status()
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check to AI model failed: {e}")
        return jsonify({"error": "Failed to reach AI model"}), 502


@app.route('/blur', methods=['POST'])
def blur():
    # Check image has been uploaded (more checks done by the AI model)
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Send image to the AI model
    image = request.files['image']
    response = requests.post(AI_CONTAINER_URL + "/blur", files={'image': image})

    # Return response from the AI model
    if response.status_code == 200:
        return response.content, 200, {'Content-Type': 'image/jpeg'}
    else:
        return response.json(), response.status_code


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
