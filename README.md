# Train and Deploy ByteTrack model on custom dataset with Amazon SageMaker

[ByteTrack](https://github.com/ifzhang/ByteTrack) is a simple, fast and strong multi-object tracker. In this repository, we demonstrate how to train and deploy a ByteTrack model with SageMaker from scratch on custom dataset, which consists of:
- Label and process custom dataset with SageMaker Ground Truth
- Train a ByteTrack mdoel
- Deploy a ByteTrack model
    - Batch inference
    - Real-time endpoint
    - Asynchronous endpoint

## Prerequisites
- [Create an AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/) or use the existing AWS account.
- Make sure that you have a minimum of one GPU instance (in our sample code, we use `ml.p3.16xlarge`) for the Training Job. If it is the first time you train a machine learning model on the GPU instance, you will need to [request a service quota increase for SageMaker Training Jobs]( https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html).
- [Create a SageMaker Notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html). The default volume size is 5GB, you need to update the volume size to 50GB. For IAM role, choose the existing IAM role or create a new IAM role, attach the policy of `AmazonSageMakerFullAccess` and `AmazonElasticContainerRegistryPublicFullAccess` to the chosen IAM role.
- Make sure that you have a minimum of one GPU instance (in our sample code, we use `ml.p3.2xlarge`) for Infenrece endpoint. If it is the first time you deploy a machine learning model on the GPU instance, you will need to [request a service quota increase for SageMaker Endpoints]( https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html).
- Make sure that you have a minimum of one GPU instance (in our sample code, we use `ml.p3.2xlarge`) for Processing jobs. If it is the first time you run a processing job on the GPU instance, you will need to [request a service quota increase for SageMaker Processing Jobs]( https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html).
- The region `us-east-1` is recommended.

## 1. Label data with SageMaker Ground Truth
We simulate the real business scenario to label video data. By running the notebook [data-preparation.ipynb](./data-preparation.ipynb), we can download sample video and prepare the video clips for input data for SageMaker Ground Truth. Then you can follow [Use Amazon SageMaker Ground Truth to Label Data](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-getting-started.html) to create a labeling job and label data you prepared. Note that it takes time to label all objects in all frames, to accelarate the labelling job, you can label only few objects(person) and leverage `Predict` function in SageMaker Ground Truth. For your reference, we labeled about 15 persons in each 2 frames, which took about 2 hours by leveraging `Predict` function. You can change the extraction rate in **Frame extraction** when creating a Labeling Job with SageMaker Ground Truth Console to accelerate labeling job.

## 2. Model training
Once ground truth data is ready, we can run [bytetrack-training.ipynb](bytetrack-training.ipynb) to train a model. In this session, we convert ground truth data from SageMage Ground Truth into dataset which trainable to ByteTrack, and train a model with BYOC mode.

## 3. Model deployment
After you get a trained model in model training, we provide three options for inference. When adapting tracking solution in your business, you can choose the inference option suitable for your business requirements. 
- Run batch inference with [bytetrack-batch-inference.ipynb](bytetrack-batch-inference.ipynb).
- Deploy a real-time inference endpoint with [bytetrack-inference-yolox.ipynb](bytetrack-inference-yolox.ipynb).
- Deploy an Asynchronous inference endpoint with [bytetrack-inference-yolox-async.ipynb](bytetrack-inference-yolox-async.ipynb).