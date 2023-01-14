import json
import os
import shutil
import subprocess
import sys

def copy_files(src, dest):
    src_files = os.listdir(src)
    for file in src_files:
        path = os.path.join(src, file)
        if os.path.isfile(path):
            shutil.copy(path, dest)
            
def train():
    import pprint
    
    pprint.pprint(dict(os.environ), width = 1) 
    
    model_dir = os.environ['SM_MODEL_DIR']
    log_dir = None
    
    copy_logs_to_model_dir = False
    
    try:
        log_dir = os.environ['SM_CHANNEL_LOG']
        copy_logs_to_model_dir = True
    except KeyError:
        log_dir = model_dir
        
    mot_data_dir = os.environ['SM_CHANNEL_MOT']
    pretrain_data_dir = os.environ['SM_CHANNEL_PRETRAIN']
    hyperparamters = json.loads(os.environ['SM_HPS'])
    
    try:
        batch_size = hyperparamters['batch_size']
    except KeyError:
        batch_size = 64
    
    try:
        fp16 = hyperparamters['fp16']
    except KeyError:
        fp16 = 1
    
    try:
        occupy = hyperparamters['occupy']
    except KeyError:
        occupy = 1
        
    try:
        pretrained_model = hyperparamters['pretrained_model']
    except KeyError:
        print('Error in pretrained_model')
        pretrained_model = 'yolox_x.pth'
        
    try:
        num_classes = hyperparamters['num_classes']
    except KeyError:
        num_classes = 1
        
    try:
        depth = hyperparamters['depth']
    except KeyError:
        depth = 1.33
        
    try:
        width = hyperparamters['width']
    except KeyError:
        width = 1.25
        
    try:
        input_size_h = hyperparamters['input_size_h']
    except KeyError:
        input_size_h = 800
    
    try:
        input_size_w = hyperparamters['input_size_w']
    except KeyError:
        input_size_w = 1440
    
    try:
        test_size_h = hyperparamters['test_size_h']
    except KeyError:
        test_size_h = 800
    
    try:
        test_size_w = hyperparamters['test_size_w']
    except KeyError:
        test_size_w = 1440
    
    try:
        random_size_h = hyperparamters['random_size_h']
    except KeyError:
        random_size_h = 18
        
    try:
        random_size_w = hyperparamters['random_size_w']
    except KeyError:
        random_size_w = 32
    
    try:
        max_epoch = hyperparamters['max_epoch']
    except KeyError:
        max_epoch = 80
    
    try:
        print_interval = hyperparamters['print_interval']
    except KeyError:
        print_interval = 20
        
    try:
        eval_interval = hyperparamters['eval_interval']
    except KeyError:
        eval_interval = 5
    
    try:
        test_conf = hyperparamters['test_conf']
    except KeyError:
        test_conf = 0.001
    
    try:
        nmsthre = hyperparamters['nmsthre']
    except KeyError:
        nmsthre = 0.7
    
    try:
        basic_lr_per_img = hyperparamters['basic_lr_per_img']
    except KeyError:
        basic_lr_per_img = 0.001 / 64.0
    
    try:
        no_aug_epochs = hyperparamters['no_aug_epochs']
    except KeyError:
        no_aug_epochs = 10
    
    try:
        warmup_epochs = hyperparamters['warmup_epochs']
    except KeyError:
        warmup_epochs = 1
    
    try:
        infer_device = hyperparamters['infer_device']
    except KeyError:
        infer_device = 'cuda'
    
    gpus_per_host = int(os.environ['SM_NUM_GPUS'])
    
    train_cmd = f"""
cd /ByteTrack && python3 tools/train.py \
--batch_size {batch_size} \
--dist-backend 'nccl' \
--devices {gpus_per_host} \
--local_rank 0 \
--num_machines 1 \
--machine_rank 0 \
--num_classes {num_classes} \
--depth {depth} \
--width {width} \
--input_size_h {input_size_h} \
--input_size_w {input_size_w} \
--test_size_h {test_size_h} \
--test_size_w {test_size_w} \
--random_size_h {random_size_h} \
--random_size_w {random_size_w} \
--max_epoch {max_epoch} \
--print_interval {print_interval} \
--eval_interval {eval_interval} \
--test_conf {test_conf} \
--nmsthre {nmsthre} \
--basic_lr_per_img {basic_lr_per_img} \
--no_aug_epochs {no_aug_epochs} \
--warmup_epochs {warmup_epochs} \
--model_dir {model_dir} \
--data_dir {mot_data_dir} \
--ckpt {pretrain_data_dir}/{pretrained_model} \
--infer_device {infer_device} \
"""
    
    if fp16 == 1:
        train_cmd += "--fp16"
    
    if occupy == 1:
        train_cmd += "-o"
    
    print("--------Begin Model Training Command----------")
    print(train_cmd)
    print("--------End Model Training Comamnd------------")
    exitcode = 0
    try:
        process = subprocess.Popen(
            train_cmd,
            encoding='utf-8', 
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        while True:
            if process.poll() != None:
                break

            output = process.stdout.readline()
            if output:
                print(output.strip())
        
        out, err = process.communicate()
        print(err)
        print(out)
        
        exitcode = process.poll() 
        print(f"exit code:{exitcode}")
        exitcode = 0 
    except Exception as e:
        print("train exception occured", file=sys.stderr)
        exitcode = 1
        print(str(e), file=sys.stderr)
    finally:
        if copy_logs_to_model_dir:
            copy_files(log_dir, model_dir)
    
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(exitcode)

if __name__ == "__main__":
    train()