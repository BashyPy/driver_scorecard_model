FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8501

COPY . /app

#ENTRYPOINT [ "python" ]

HEALTHCHECK --interval=5s --timeout=3s CMD curl -f http://localhost:8501 || nc -zv localhost 8501 || exit 1

CMD ["streamlit", "run", "main.py"]
