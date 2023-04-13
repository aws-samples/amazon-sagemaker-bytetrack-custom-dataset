# Train and Deploy ByteTrack model on custom dataset with Amazon SageMaker

[ByteTrack](https://github.com/ifzhang/ByteTrack) is a simple, fast and strong multi-object tracker. In this repository, we demonstrate how to train and deploy a ByteTrack model with SageMaker from scratch on custom dataset. which consists of:
- Label and process custom dataset with SageMaker Ground Truth
- Train a ByteTrack mdoel
- Deploy a ByteTrack model
    - Batch inference
    - Real-time endpoint on p3.2xlarge
    - Asynchronous endpoint on p3.2xlarge

## 1. Label data with SageMaker Ground Truth
Upload video clips and follow [Use Amazon SageMaker Ground Truth to Label Data](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-getting-started.html) to create a labeling job and label data you prepared. Then run the notebook [data_labeling.ipynb](./data_labeling.ipynb).

## 2. Model training
Once ground truth data is ready, we can run [bytetrack-training.ipynb](bytetrack-training.ipynb) to train a model. In this session, we convert ground truth data from SageMage Ground Truth into dataset which trainable to ByteTrack.

## 3. Model deployment
By using SageMaker SDK in [bytetrack-inference-yolox.ipynb](bytetrack-inference-yolox.ipynb), we can deploy a real-time inference endpoint. To run a batch inference, we can run [bytetrack-batch-inference.ipynb](bytetrack-batch-inference.ipynb). You also can deploy an asychronous inference endpoint by running [bytetrack-inference-yolox-async.ipynb](bytetrack-inference-yolox-async.ipynb).