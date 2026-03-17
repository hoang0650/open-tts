FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/api.py"]