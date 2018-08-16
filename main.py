#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import argparse
import train
import evaluate
import hparams_helper

def add_arguments(parser):
    parser.add_argument("--mode", choices=["train","test","infer"], type=str,
                        default="train", help="run mode train|dev|test")
    parser.add_argument("--train_file", type=str, default="data/train.csv",
                        help="train dataset file, require csv format")
    parser.add_argument("--hparams_default", type=str,
                        default="hparams_default.json",
                        help="default hparams json file")

def main(argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    hparams = hparams_helper.create_default_hparams(args.hparams_default)
    hparams = hparams_helper.merge_hparams(hparams, args)
    print("detect tensorflow version {}".format(tf.__version__))
    
    if args.mode == "train":
        train.train(hparams)
    elif args.mode == "test":
        evaluate.evaluate(hparams)
    else:
        print("error parameters")

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
