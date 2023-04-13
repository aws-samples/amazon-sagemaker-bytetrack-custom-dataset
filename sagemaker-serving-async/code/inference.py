import io
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import torchvision
import json
import time
import os
import tempfile
import logging

#import sys
#sys.path.append("/opt/ml/model/mycode")

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils.visualize import plot_tracking

# This code will be loaded on each worker separately..
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'

device = torch.device("cuda")

def model_fn(model_dir):
    try:
        model = torch.load(f"{model_dir}/model.pth", map_location=device)
    except Execept as e:
        print(str(e))
    return model

def transform_fn(model, request_body, content_type, accept):
    interval = int(os.environ.get('FRAME_INTERVAL', 1))
    input_width = int(os.environ.get('INPUT_WIDTH', 0))
    input_height = int(os.environ.get('INPUT_HEIGHT', 0))
    #batch_size = int(os.environ.get('BATCH_SIZE', 1))

    f = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_id = 0
    
    # Define ByteTrack
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    tracker = BYTETracker(
        frame_rate=30,
        track_thresh=0.5,
        track_buffer=30,
        mot20=False,
        match_thresh=0.8
    )
    timer = Timer()
    results = []

    while cap.isOpened():
        success, frame = cap.read()
        
        timer.tic()
        if not success:
            cap.release()
            break

        if frame_id % interval == 0:
            print(f'frame_id: {frame_id}')
            batch_inputs = input_preproc(frame, input_height, input_width)  # returns 4D tensor
            batch_outputs = predict(batch_inputs, model)
            batch_predictions = output_process(batch_outputs)
            
            if batch_predictions[0] is not None:
                online_targets = tracker.update(
                    torch.as_tensor(batch_predictions[0]),
                    [height, width],
                    (input_height, input_width)
                )
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1"
                        )
                timer.toc()
                online_im = plot_tracking(
                    frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = frame
            
            if (frame_id/interval) % 20 == 0:
                print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        
        frame_id += 1
    else:
        raise Exception("Failed to open video '%s'!.." % tfile.name)
    
    logger.info(">>> Length of final predictions: %d" % len(results))
    return json.dumps(results)

def predict(inputs, model):
    with torch.no_grad():
        model = model.to(device)
        input_data = inputs.to(device)
        model.eval()
        outputs = model(input_data)

    return outputs

def input_preproc(open_cv_image, input_height, input_width):
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
    
    img, ratio = preproc(open_cv_image, (input_height, input_width), rgb_means, std)
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    return img

def output_process(prediction_output, accept=JSON_CONTENT_TYPE):
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
    return prediction_output