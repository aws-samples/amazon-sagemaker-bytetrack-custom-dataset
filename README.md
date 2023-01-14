# Train and Deploy ByteTrack with Amazon SageMaker

[ByteTrack](https://github.com/ifzhang/ByteTrack) is a simple, fast and strong multi-object tracker. In this project, we demonstrate how to deliver an end-to-end ByteTrack model, including:
- Label data with SageMaker Ground Truth
- Train a ByteTrack mdoel
- Deploy an ByteTrack model
    - Batch inference
    - Deploy a real-time endpoint on p3.2xlarge
    - Deploy a real-time endpoint on Inf1

## 1. Label data with SageMaker Ground Truth
Upload video clips and follow [Use Amazon SageMaker Ground Truth to Label Data](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-getting-started.html) to create a labeling job and label data you prepared. Then run the notebook [data_labeling.ipynb](./data_labeling.ipynb).

## 2. Train a model
Once ground truth data is ready, we can run [bytetrack-training.ipynb](bytetrack-training.ipynb) to train a model. In this session, we convert ground truth data from SageMage Ground Truth into dataset which trainable to ByteTrack.

## 3. Deploy a model
By using SageMaker SDK in [bytetrack-inference-yolox.ipynb](bytetrack-inference-yolox.ipynb), we can deploy a real-time inference endpoint. To run a batch inference, we can run [bytetrack-batch-inference.ipynb](bytetrack-batch-inference.ipynb).