import unittest
from io import BytesIO
from unittest.mock import patch, MagicMock

import requests

from ApiGateway.api_gateway import app


class APIGatewayTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/hc')
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', response.json)
        self.assertIn('uptime', response.json)

    @patch('ApiGateway.api_gateway.requests.get')
    def test_hc_model_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response

        response = self.app.get('/hc_model')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})

    @patch('ApiGateway.api_gateway.requests.get')
    def test_hc_model_failure(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException

        response = self.app.get('/hc_model')
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json, {"error": "Failed to reach AI service"})

    @patch('ApiGateway.api_gateway.requests.post')
    def test_blur_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image_data"
        mock_post.return_value = mock_response

        data = {
            'image': (BytesIO(b"fake_image_data"), 'test.jpg')
        }
        response = self.app.post('/blur', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"fake_image_data")

    def test_blur_no_image(self):
        response = self.app.post('/blur')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "No image file provided"})

    @patch('ApiGateway.api_gateway.requests.post')
    def test_blur_ai_service_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "AI model error"}
        mock_post.return_value = mock_response

        data = {
            'image': (BytesIO(b"fake_image_data"), 'test.jpg')
        }
        response = self.app.post('/blur', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {"error": "AI model error"})


if __name__ == '__main__':
    unittest.main()
