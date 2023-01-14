#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self,
                num_classes=1,
                depth=1.33,
                width=1.25,
                exp_name=os.path.split(os.path.realpath(__file__))[1].split(".")[0],
                train_ann="train.json",
                val_ann="val_half.json",
                input_size=(800, 1440),
                test_size=(800, 1440),
                random_size=(18, 32),
                max_epoch=80,
                print_interval=20,
                eval_interval=5,
                test_conf=0.1,
                nmsthre=0.7,
                no_aug_epochs=10,
                basic_lr_per_img=0.001/64.0,
                warmup_epochs=1,
                output_dir='/tmp/ml/logs',
                data_dir='',
                infer_device='cuda'
            ):
        super(Exp, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.width = width
        self.exp_name = exp_name
        self.train_ann = train_ann
        self.val_ann = val_ann
        self.input_size = input_size
        self.test_size = test_size
        self.random_size = random_size
        self.max_epoch = max_epoch
        self.print_interval = print_interval
        self.eval_interval = eval_interval
        self.test_conf = test_conf
        self.nmsthre = nmsthre
        self.no_aug_epochs = no_aug_epochs
        self.basic_lr_per_img = basic_lr_per_img
        self.warmup_epochs = warmup_epochs
        
        self.infer_device = infer_device
        
        self.output_dir = output_dir
        self.train_dir = data_dir
        self.var_dir = data_dir

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=self.train_dir,
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=self.var_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name='train',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
