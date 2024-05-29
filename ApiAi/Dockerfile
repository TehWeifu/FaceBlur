# Dockerfile para IA Container

FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
COPY ia_container.py ia_container.py
COPY models models
COPY imgs imgs

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install -r requirements.txt

CMD ["python", "ia_container.py"]
