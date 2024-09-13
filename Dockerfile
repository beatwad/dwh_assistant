FROM python:3.11-slim-bookworm

WORKDIR /app

COPY . /app

VOLUME /app/ap_template

RUN pip install -r requirements.txt

CMD ["python", "run.py"]
