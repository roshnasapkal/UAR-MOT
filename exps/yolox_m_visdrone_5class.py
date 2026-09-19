#!/usr/bin/env python3

import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):

    def __init__(self):
        super(Exp, self).__init__()

        # Architecture
        self.depth = 0.67
        self.width = 0.75
        self.num_classes = 5

        # Reproducibility
        self.seed = 42

        self.exp_name = "yolox_m_visdrone_5class"

        # Dataset
        self.data_dir = "/content/UAR-MOT_detector_data"

        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # Resolution
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (14, 26)

        # Optimization: YOLOX 0.1.0 defaults unless noted
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"

        self.warmup_epochs = 5
        self.warmup_lr = 0

        self.min_lr_ratio = 0.05

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.ema = True

        # Fixed fine-tuning schedule
        self.max_epoch = 100
        self.no_aug_epochs = 15

        # Runtime / evaluation
        self.data_num_workers = 2
        self.print_interval = 10
        self.eval_interval = 10

        # Evaluation defaults
        self.test_conf = 0.01
        self.nmsthre = 0.65
