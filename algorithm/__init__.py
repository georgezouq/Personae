# coding=utf-8

import tensorflow as tf

from os import environ

environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"

config = tf.ConfigProto(
    device_count={"CPU": 4},
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=4,
    log_device_placement=True
)

config.gpu_options.allow_growth = True
config.allow_soft_placement = True
