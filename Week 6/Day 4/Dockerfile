FROM python:3.12-slim

WORKDIR /app
COPY ["Week 5", "/app/Week 5"]
COPY ["Week 6", "/app/Week 6"]

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "/app/Week 6/Day 2/service.py", "--host", "0.0.0.0", "--port", "8080"]
