## The Makefile includes instructions on environment setup and lint tests
# Create and activate a virtual environment
# Install dependencies in requirements.txt
# Dockerfile should pass hadolint
# app.py should pass pylint
# (Optional) Build a simple integration test

setup:
	# Create python virtualenv & source it
	#source ~/venv/bin/activate
	python3 -m venv ~/venv

install:
	# This should be run from inside a virtualenv
	pip3 install --no-cache-dir -r requirements.txt
	#wget -O ./hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.1-beta/hadolint-Darwin-x86_64 &&\
	#	chmod +x ./hadolint

test:
	# Additional, optional, tests could go here
	#python -m pytest -vv --cov=myrepolib tests/*.py
	#python -m pytest --nbval notebook.ipynb

lint:
	# See local hadolint install instructions:   https://github.com/hadolint/hadolint
	# This is linter for Dockerfiles
	#hadolint Dockerfile
	# This is a linter for Python source code linter: https://www.pylint.org/
	# This should be run from inside a virtualenv
	pylint --disable=R,C,W1203 main.py
	pylint --disable=R,C,W1203 users.py
	pylint --disable=R,C,W1203 error_logger.py
	pylint --disable=R,C,W1203 pydantic_models.py
	pylint --disable=R,C,W1203 users.py
	pylint --disable=R,C,W1203 app.py

all: install lint test
