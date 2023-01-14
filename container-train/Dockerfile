FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker

RUN git clone https://github.com/ifzhang/ByteTrack /ByteTrack \
    && cd /ByteTrack \
    && pip3 install -r requirements.txt \
    && python3 setup.py develop \
    && pip3 install cython \
    && pip3 install cython_bbox \
    && pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

##########################################################################################
# SageMaker requirements
##########################################################################################
COPY resources/train.py /opt/ml/code/train.py
COPY resources/ByteTrack/train.py /ByteTrack/tools/train.py
COPY resources/ByteTrack/trainer.py /ByteTrack/yolox/core/trainer.py
COPY resources/ByteTrack/data_loader.py /ByteTrack/tools/data_loader.py
COPY resources/ByteTrack/mot.py /ByteTrack/yolox/data/datasets/mot.py

ENV SAGEMAKER_PROGRAM train.py
