FROM nvidia/cuda:12.2.2-base-ubuntu22.04

WORKDIR /app

RUN apt update

RUN apt-get install -y python3 python3-pip curl ffmpeg libsm6 libxext6 sed -y
RUN apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY wheels wheels
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY models models
RUN curl -L https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/shape_predictor_68_face_landmarks.dat > models/shape_predictor_68_face_landmarks.dat
COPY asserts asserts
RUN mkdir asserts/inference_result
RUN curl -L https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/output_graph.pb > asserts/output_graph.pb
RUN curl -L https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/pretrained_lipsick.pth > asserts/pretrained_lipsick.pth

COPY app.py app.py
COPY server.py server.py
COPY compute_crop_radius.py compute_crop_radius.py
COPY sync_batchnorm sync_batchnorm
COPY utils utils
COPY config config
COPY inference.py inference.py

CMD ["python3", "-u", "server.py"]
