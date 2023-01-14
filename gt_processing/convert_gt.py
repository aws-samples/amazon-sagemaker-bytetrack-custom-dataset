import os
import json
import boto3
from distutils.dir_util import copy_tree
import shutil
import argparse

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

ANN_DIR = "annotations/worker-response"
MANIFEST_PATH = "manifests/output/output.manifest"
    
GT_DIR = "/opt/ml/processing/gt"
INPUT_DIR = "/opt/ml/processing/input"
MOT_DIR = "/opt/ml/processing/output"

def read_json_from_s3(bucket_name, key):
    content_object = s3.Object(bucket_name, key)
    content = content_object.get()['Body'].read().decode('utf-8')
    return json.loads(content)

def read_json_from_file(dir_name, key):
    with open(os.path.join(dir_name, key), 'r') as f:
        return json.loads(f.read())

def split_s3uri(s3uri):
    s3uri = s3uri.replace('s3://', '')
    bucket_name = s3uri.split('/')[0]
    key = s3uri.replace(f'{bucket_name}/', '')
    return bucket_name, key

def create_ann_file(seq_name, class_map, split, file_name='det'):
    det_ann = []
    for path, subdirs, files in os.walk(os.path.join(GT_DIR, ANN_DIR)):
        for name in files:
            file_path = os.path.join(path, name)
            if 'frame' not in file_path:
                continue
            
            with open(file_path, 'r') as f:
                ann_json = json.loads(f.read())
            bboxes = ann_json['trackingAnnotations']['boundingBoxes']
            frame_num = ann_json['trackingAnnotations']['frameNumber']

            for bbox in bboxes:
                top, left, width, height = bbox['top'], bbox['left'], bbox['width'], bbox['height']
                label_attr = bbox['labelCategoryAttributes']
                ratio_vis = label_attr['ratio_vis'] if 'ratio_vis' in label_attr else 1
                conf_score = label_attr['conf_score'] if 'conf_score' in label_attr else 1
                
                if file_name == 'det':
                    ann = [frame_num, -1, top, left, width, height, conf_score, -1, -1]
                else:
                    obj_id = bbox['objectName'].split(':')[-1]
                    class_name = bbox['objectName'].split(':')[0]
                    ann = [frame_num, obj_id, top, left, width, height, conf_score, class_map[class_name], ratio_vis]
                det_ann.append(','.join(str(item) for item in ann))
    
    det_dir = os.path.join(MOT_DIR, split, seq_name, file_name)
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)
    
    with open(os.path.join(det_dir, f'{file_name}.txt'), 'w') as f:
        f.write("\n".join(str(item) for item in det_ann))

def convert_seq_ann(source_ref, class_map, split):
    # Read source ref
    source_ref_bucket, source_ref_key = split_s3uri(source_ref)
    json_source_ref = read_json_from_s3(source_ref_bucket, source_ref_key)
    
    # Read seqeunce name
    seq_name = json_source_ref['prefix'].split('/')[-2]
    
    seq_dir = os.path.join(INPUT_DIR, seq_name)
    seq_dest = os.path.join(MOT_DIR, split, seq_name, 'img1')
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)
    
    copy_tree(seq_dir, seq_dest)
    
    for file_name in ['det', 'gt']:
        create_ann_file(seq_name, class_map, split, file_name=file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--test-ratio", type=float, default=0)
    parser.add_argument("--labeling-job-name", type=str, default=0)
    
    args, _ = parser.parse_known_args()
    
    seq_list = []
    # Read output manifest
    with open(os.path.join(GT_DIR, MANIFEST_PATH), 'r') as f:
        lines = f.readlines()
        for line in lines:
            seq_manifest = json.loads(line)
            seq_list.append(seq_manifest)
            
    
    seq_num = len(seq_list)
    print(f'the number of sequences: {seq_num}')
    train_num = int(seq_num*args.train_ratio)
    val_num = int(seq_num*args.val_ratio)
    test_num = int(seq_num*args.test_ratio)
    
    train_cnt = 0
    val_cnt = 0
    for seq_json in seq_list:
        source_ref = seq_json['source-ref']
        id_to_class = seq_json[f'{args.labeling_job_name}-ref-metadata']['class-map']
        
        class_to_id = {v: k for k, v in id_to_class.items()}
        
        if train_cnt < train_num:
            convert_seq_ann(source_ref, class_to_id, split='train')
            train_cnt += 1
        elif val_cnt < val_num:
            convert_seq_ann(source_ref, class_to_id, split='val')
            val_cnt += 1
        else:
            convert_seq_ann(source_ref, class_to_id, split='test')