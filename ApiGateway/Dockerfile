# Dockerfile para API Gateway

FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
COPY api_gateway.py api_gateway.py

RUN pip install -r requirements.txt

CMD ["python", "api_gateway.py"]
