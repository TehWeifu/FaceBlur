# Dockerfile para IA Container
FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ia_container.py ia_container.py

CMD ["python", "ia_container.py"]
