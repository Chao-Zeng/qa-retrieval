# -*- coding:utf-8 -*-

import tensorflow as tf
import json
import os, sys

def create_default_hparams(filename):
    if not os.path.exists(filename):
        print("config file {} not exist".format(filename))
        sys.exit(-1)
    with open(filename) as f:
        params_default = json.load(f)        
    hparams = tf.contrib.training.HParams(**params_default)
    return hparams

def merge_hparams(hparams, args):
    "override hparams by command line args"
    return hparams
