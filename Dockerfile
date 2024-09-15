FROM python:3.11-slim-bookworm

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* 

WORKDIR /app/ap_template

EXPOSE 5000
EXPOSE 5432

CMD ["python", "run.py"]
