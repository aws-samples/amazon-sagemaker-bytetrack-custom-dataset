# Original Copyright (c) 2021 Yifu Zhang. Licensed under MIT License.
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
#from yolox.exp import get_exp

import argparse
import random
import warnings

from data_loader import Exp
import os

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size"
    )
    
    parser.add_argument(
        "--devices", default=None, type=int, help="device for training"
    )
    
    parser.add_argument(
        "--infer_device", default=None, type=str, help="device for inference"
    )
    
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--exp_file",
        default=None,
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        type=str,
        help="checkpoint file"
    )
    
    parser.add_argument(
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    
    parser.add_argument(
        '--num_classes',
        default=1,
        type=int,
        help="number of classes"
    )
    
    parser.add_argument(
        '--depth',
        default=1.33,
        type=float,
        help=""
    )
    
    parser.add_argument(
        '--width',
        default=1.25,
        type=float,
        help=""
    )
    
    parser.add_argument(
        '--input_size_h',
        default=800,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--input_size_w',
        default=1440,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--test_size_h',
        default=800,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--test_size_w',
        default=1440,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--random_size_h',
        default=18,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--random_size_w',
        default=32,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--max_epoch',
        default=80,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--print_interval',
        default=20,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--eval_interval',
        default=5,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--test_conf',
        default=0.001,
        type=float,
        help=""
    )
    
    parser.add_argument(
        '--nmsthre',
        default=0.7,
        type=float,
        help=""
    )
    
    parser.add_argument(
        '--basic_lr_per_img',
        default=0.001 / 64.0,
        type=float,
        help=""
    )
    
    parser.add_argument(
        '--no_aug_epochs',
        default=10,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--warmup_epochs',
        default=1,
        type=int,
        help=""
    )
    
    parser.add_argument(
        '--train_ann',
        default="train.json",
        type=str,
        help=""
    )
    
    parser.add_argument(
        '--val_ann',
        default="train.json",
        type=str,
        help=""
    )
    
    parser.add_argument(
        '--data_dir',
        default="",
        type=str,
        help=""
    )
    
    parser.add_argument(
        '--model_dir',
        default="",
        type=str,
        help=""
    )
    
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    
    exp = Exp(
        num_classes=args.num_classes,
        depth=args.depth,
        width=args.width,
        exp_name=os.path.split(os.path.realpath(__file__))[1].split(".")[0],
        train_ann=args.train_ann,
        val_ann=args.val_ann,
        input_size=(args.input_size_h, args.input_size_w),
        test_size=(args.test_size_h, args.test_size_w),
        random_size=(args.random_size_h, args.random_size_w),
        max_epoch=args.max_epoch,
        print_interval=args.print_interval,
        eval_interval=args.eval_interval,
        test_conf=args.test_conf,
        nmsthre=args.nmsthre,
        no_aug_epochs=args.no_aug_epochs,
        basic_lr_per_img=args.basic_lr_per_img,
        warmup_epochs=args.warmup_epochs,
        output_dir=args.model_dir,
        data_dir=args.data_dir,
        infer_device=args.infer_device
    )
    
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args),
    )