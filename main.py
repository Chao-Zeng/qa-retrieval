#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import argparse

def add_arguments(parser):
    parser.add_argument("--mode", choices=["train","dev","test"], type=str,
                        default="train", help="run mode train|dev|test")
    parser.add_argument("--train_file", type=str, default="data/train.csv",
                        help="train dataset file, require csv format")

def main(argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
