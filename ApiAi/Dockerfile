# Dockerfile para IA Container

FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
COPY ia_container.py ia_container.py

RUN pip install -r requirements.txt

CMD ["python", "ia_container.py"]
