FROM tensorflow/tensorflow:2.19.0-gpu

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get install -y python3.12 python3.12-dev python3.12-venv && \
    apt-get install -y python3-pip

RUN python3.12 -m venv /opt/venv
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install tensorflow[and-cuda]==2.19.0

ENV PATH="/opt/venv/bin:$PATH"

RUN python --version && python -c "import tensorflow as tf; print(tf.__version__)"

# One resolve with tensorflow pinned, so numpy<2 is honoured by every package
# rather than forced afterwards. numpy must come as a wheel: a backtrack onto an
# old numpy sdist cannot build on Python 3.12. gputools is left out: it is only
# StarDist's optional OpenCL acceleration, and its pyopencl fights TF's numpy pin.
RUN pip install --only-binary=numpy \
    "tensorflow[and-cuda]==2.19.0" stardist edt \
    "arkitekt[all]>=3.2.0" "mikro>=5.0.0" "numpy<2"

RUN mkdir /app
WORKDIR /app
# The whole project: the app target may be any module in it, not only app.py.
COPY . /app

# Bake both pretrained models into the image
RUN python init.py
