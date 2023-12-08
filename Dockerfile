FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8080

COPY . /app

ENTRYPOINT [ "python" ]

CMD ["main.py"]
