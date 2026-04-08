# Dockerfile
# Packages the entire project into one container
# Judges and HF Spaces use this to run your project

# ─────────────────────────────────────────
# BASE IMAGE
# Use official Python 3.11 slim version
# Slim = smaller size, faster to build
# ─────────────────────────────────────────
FROM python:3.11-slim

# ─────────────────────────────────────────
# SET WORKING DIRECTORY
# All commands run from /app inside container
# ─────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────
# COPY REQUIREMENTS FIRST
# Docker caches this layer
# Only reinstalls if requirements.txt changes
# ─────────────────────────────────────────
COPY requirements.txt .

# ─────────────────────────────────────────
# INSTALL ALL LIBRARIES
# ─────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────
# COPY ALL PROJECT FILES
# ─────────────────────────────────────────
COPY env/ ./env/
COPY inference.py .

RUN touch env/__init__.py

# ─────────────────────────────────────────
# ENVIRONMENT VARIABLES
# Default values — overridden in HF Spaces
# ─────────────────────────────────────────
ENV TASK=easy
ENV API_BASE_URL=http://127.0.0.1:8000
ENV MODEL_NAME=gpt-4o-mini
ENV HF_TOKEN=""
ENV SERVER_URL=http://127.0.0.1:8000

# ─────────────────────────────────────────
# EXPOSE PORT
# HF Spaces expects port 7860
# ─────────────────────────────────────────
EXPOSE 7860

# ─────────────────────────────────────────
# START COMMAND
# Runs the FastAPI server when container starts
# ─────────────────────────────────────────
CMD ["uvicorn", "env.app:app", "--host", "0.0.0.0", "--port", "7860"]