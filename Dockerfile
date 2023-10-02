FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install dependencies \
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /app

WORKDIR /app

COPY . /app

# wandb environment key WANDB_API_KEY
#ENV WANDB_API_KEY=YOUR_API_KEY

RUN wandb off

RUN python3 -O fast_solve.py