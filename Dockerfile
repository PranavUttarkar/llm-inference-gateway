FROM python:3.13.9-slim
WORKDIR /app
COPY gateway ./gateway
COPY tests ./tests
ENV PYTHONUNBUFFERED=1
