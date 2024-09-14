FROM python:3.11-slim-bookworm

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /usr/src/app/

COPY . /usr/src/app/

WORKDIR /usr/src/app/ap_template

EXPOSE 5000

EXPOSE 5432

CMD ["python", "run.py"]
