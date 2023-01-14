import io
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import torchvision
import json
import time

JSON_CONTENT_TYPE = 'application/json'

#device = torch.device("cpu")
device = torch.device("cuda")

def model_fn(model_dir):
    try:
        model = torch.load(f"{model_dir}/model.pth", map_location=device)
    except Execept as e:
        print(str(e))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    s = time.time()
    def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    rgb_means = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    f = io.BytesIO(request_body)
    input_image = Image.open(f).convert("RGB")
    open_cv_image = np.array(input_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    
    img, ratio = preproc(open_cv_image, (800, 1440), rgb_means, std)
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    print('input_fn:', time.time()-s)
    return img

def predict_fn(input_data, model):
    s = time.time()
    with torch.no_grad():
        outputs = model(input_data)
    
    print('inference time: ', time.time()-s)
    return outputs

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    s = time.time()
    def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
            )

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections.tolist()
            else:
                output[i] = torch.cat((output[i], detections)).tolist()

        return output
    
    prediction_output = postprocess(prediction_output, 1, 0.7, 0.45)
    
    print('output_fn: ', time.time()-s)
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    raise Exception('Requested unsupported ContentType in Accept: ' + accept)