# FROM python:3.11
FROM python:3.10


ENV DEBIAN_FRONTEND=noninteractive

# System deps (add more if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libsm6 libxext6 libxrender-dev ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /workspace/requirements.txt

# Create user to avoid running as root
ARG NB_USER=dev
ARG NB_UID=1000
RUN useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER}
USER ${NB_USER}
ENV HOME=/home/${NB_USER}
WORKDIR /workspace



# Expose ports for jupyterlab and streamlit
EXPOSE 8888 8501

# Default command opens jupyterlab (tokenless for this example — see .env for production)
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' --notebook-dir=/workspace"]
