FROM python:3.11

WORKDIR /usr/src/label_extended_function

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt
