import os
import argparse
import json
from pathlib import Path
import time
import datetime
from distutils.dir_util import copy_tree

INPUT_DIR = "/opt/ml/processing/input/data"
OUTPUT_DIR = "/opt/ml/processing/output"

#INPUT_DIR = "/home/ec2-user/SageMaker/amazon-sagemaker-bytetrack/container-batch-inference/output/track_vis/2022_12_01_08_26_02/mot"
#OUTPUT_DIR = "output"

MAP_ID2CLASS = {
    "0": "pedestrian",
    "1": "car",
    "2": "bicycle"
}

MAP_CLASS2ID = {
    "pedestrian": "0",
    "car": "1",
    "bicycle": "2"
}

def convert_ann(tracker_list):
    annotations_list = []
    
    for tracker in tracker_list:
        annotations_list.append({
            "height": tracker["height"],
            "width": tracker["width"],
            "top": tracker["top"],
            "left": tracker["left"],
            "class-id": MAP_CLASS2ID[tracker["label"]],
            "label-category-attributes": tracker["labelCategoryAttributes"],
            "object-id": tracker["objectId"],
            "object-name": tracker["objectName"]
        })
        
    return annotations_list

def convert_gt(seq, seq_id, job_name):
    gt_path = os.path.join(INPUT_DIR, seq, 'gt/gt.txt')
    sm_gt_dir = f"annotations/worker-response/iteration-1/{seq_id}"
    sm_gt_dir = os.path.join(OUTPUT_DIR, job_name, sm_gt_dir)
    if not os.path.exists(sm_gt_dir):
        os.makedirs(sm_gt_dir)
    
    label_list = []
    
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        frame_id = -1
        tracker_list = []
        for line in lines:
            data_list = line.replace('\n', '').split(',')
            data_list = [float(i) for i in data_list]
            if data_list[0] != frame_id:
                if len(tracker_list) > 0:
                    # save trackers
                    frame_name = f"frame_{frame_id:04}.jpeg"
                    file_path = os.path.join(sm_gt_dir, f"{frame_name}.json")
                    
                    frame_ann = {
                        'trackingAnnotations': {
                            'frameNumber': frame_id,
                            'frame': frame_name,
                            'boundingBoxes': tracker_list,
                            'frameAttributes': {}
                        }
                    }
                    
                    with open(file_path, 'w') as f:
                        json.dump(frame_ann, f)
                    
                    label_list.append({
                        "annotations": convert_ann(tracker_list),
                        "frame-no": frame_id,
                        "frame": frame_name,
                        "frame-attributes": {}
                    })
                    
                    tracker_list = []
                frame_id += 1
            else:
                label_name = MAP_ID2CLASS[str(data_list[-2])] if str(data_list[-2]) in MAP_ID2CLASS else "unknown"
                obj_id = int(data_list[1])
                tracker_list.append({
                    "label": label_name,
                    "objectId": obj_id,
                    "objectName": f"{label_name}:{obj_id}",
                    "labelCategoryAttributes": {"conf_score": data_list[6], "ratio_vis": "1"},
                    "width": round(data_list[4]),
                    "top": round(data_list[2]),
                    "height": round(data_list[5]),
                    "left": round(data_list[3])
                })
    
    # Create seqlab
    seqlab_dir = os.path.join(OUTPUT_DIR, job_name, f"annotations/consolidated-annotation/output/{seq_id}")
    
    if not os.path.exists(seqlab_dir):
        os.makedirs(seqlab_dir)
    
    with open(os.path.join(seqlab_dir, 'SeqLabel.json'), 'w') as f:
        json.dump({"tracking-annotations": label_list}, f)
    
                
def convert_img(seq, seq_id, job_name, input_s3uri, gt_s3uri):
    img_dir = os.path.join(INPUT_DIR, seq, 'img1')
    images = Path(img_dir).glob('*.jpeg')
    
    images = [str(img_path).split('/')[-1] for img_path in images]
    img_num = len(images)
    
    time_sec = int(time.time())
    
    src_ref_name = f"{seq}-sequence-{time_sec}.json"
    src_ref_prefix = f"{gt_s3uri}/{job_name}/inputs/{seq}"
    src_ref_path = f"s3://{src_ref_prefix}/{src_ref_name}"
    
    src_ref = {
        "seq-no": seq_id,
        "prefix": src_ref_prefix,
        "number-of-frames": img_num,
        "frames": [{"frame-no": int(img_name.replace('.jpeg', '').split('_')[-1]), "frame": img_name} for img_name in images]
    }
    
    
    seq_dest_dir = os.path.join(OUTPUT_DIR, job_name, 'inputs', seq)
    if not os.path.exists(seq_dest_dir):
        os.makedirs(seq_dest_dir)
    copy_tree(img_dir, seq_dest_dir)
    
    # Create sequence reference file
    with open(os.path.join(seq_dest_dir, src_ref_name), 'w') as f:
        json.dump(src_ref, f)
    
    seqlab_path = f"annotations/consolidated-annotation/output/{seq_id}/SeqLabel.json"
    
    return {
        "source-ref": src_ref_path,
        f"{job_name}-ref": f"s3://{gt_s3uri}/{job_name}/{seqlab_path}",
        f"{job_name}-ref-metadata":{
            "class-map": MAP_ID2CLASS,
            "job-name": f"labeling-job/{job_name}",
            "human-annotated": "no",
            "creation-date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "type": "groundtruth/video-object-tracking"
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input-s3uri", type=str, default="sagemaker-us-east-1-822507008821/bytetrack/sagemaker/sg-ground-truth/inputs")
    parser.add_argument("--output-s3uri", type=str, default="my-bucket")
    parser.add_argument("--job_name", type=str, default="mot-bytetrack")
    
    args, _ = parser.parse_known_args()
    
    seq_list = os.listdir(INPUT_DIR)
    seq_list = [seq for seq in seq_list if os.path.isdir(os.path.join(INPUT_DIR, seq)) and not seq.startswith('.')]
    
    
    seq_info_list = []
    for seq_id, seq_name in enumerate(seq_list):
        print(f'seq_name: {seq_name}')
        convert_gt(seq_name, seq_id, args.job_name)
        seq_info = convert_img(seq_name, seq_id, args.job_name, args.input_s3uri, args.output_s3uri)
        seq_info_list.append(seq_info)
    
    # Save sequence information
    with open(os.path.join(OUTPUT_DIR, args.job_name, 'inputs', 'input.manifest'), 'w') as f:
        for line in seq_info_list:
            line_str = json.dumps(line)
            f.write(f"{line_str}\n")