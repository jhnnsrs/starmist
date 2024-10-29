FROM tensorflow/tensorflow:latest-gpu


RUN pip install stardist gputools edt 
RUN pip install "arkitekt-next[all]==0.8.24"
RUN pip install "numpy<2"

RUN mkdir /workspace
COPY . /workspace
WORKDIR /workspace
