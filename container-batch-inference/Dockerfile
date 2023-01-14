FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker

##########################################################################################
# SageMaker requirements
##########################################################################################
RUN git clone https://github.com/ifzhang/ByteTrack /ByteTrack \
    && cd /ByteTrack \
    && pip3 install -r requirements.txt \
    && python3 setup.py develop \
    && pip3 install cython \
    && pip3 install cython_bbox \
    && pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

COPY byte_tracker.py /ByteTrack/yolox/tracker/byte_tracker.py

ENV WORKDIR /

ENTRYPOINT ["python3"]
