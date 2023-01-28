FROM python:3.8.6-slim

WORKDIR /home/work

ADD requirements.txt /home/work

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# RUN useradd -ms /bin/bash bot

# USER bot