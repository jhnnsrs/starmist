# Start with the TensorFlow 2.11 GPU image
FROM tensorflow/tensorflow:2.19.0-gpu

# Install dependencies for Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common  && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv && \
    apt-get install -y python3-pip

# Set up a virtual environment with Python 3.11
RUN python3.11 -m venv /opt/venv

# Activate the virtual environment and install TensorFlow for Python 3.10
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install tensorflow[and-cuda]==2.19.0

# Update the PATH to use the new Python environment as default
ENV PATH="/opt/venv/bin:$PATH"

# Verify installation
RUN python --version && python -c "import tensorflow as tf; print(tf.__version__)"

# The rest of your Dockerfile can follow, including any additional dependencies or commands.


RUN pip install stardist gputools edt 
RUN pip install "arkitekt-next[all]==0.8.82"
RUN pip install "numpy<2"

RUN mkdir /workspace
COPY . /workspace
WORKDIR /workspace

RUN python init.py