# api_gateway.py: API Gateway for the Blur API. Receives image files and forwards them to the AI model for processing.

import logging
import os
from time import time

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI service URL
AI_SERVICE_HOST = os.getenv("AI_SERVICE_HOST", default="http://localhost:5001")

ts_server_start = time()


@app.route('/hc', methods=['GET'])
def hc():
    ts_server_current = time()
    return jsonify({"status": "ok", "uptime": ts_server_current - ts_server_start})


@app.route('/hc_model', methods=['GET'])
def hc_model():
    try:
        response = requests.get(f"{AI_SERVICE_HOST}/hc")
        response.raise_for_status()
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check to AI service failed: {e}")
        return jsonify({"error": "Failed to reach AI service"}), 502


@app.route('/blur', methods=['POST'])
def blur():
    # Check image has been uploaded (more checks done by the AI model)
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image = request.files['image']

    try:
        response = requests.post(f"{AI_SERVICE_HOST}/blur", files={'image': image})
        if response.status_code == 200:
            return response.content, 200, {'Content-Type': 'image/jpeg'}
        else:
            logger.error(f"AI model returned an error: {response.status_code} - {response.text}")
            return response.json(), response.status_code

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to forward image to AI service: {e}")
        return jsonify({"error": "Failed to forward image to AI service"}), 502


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
