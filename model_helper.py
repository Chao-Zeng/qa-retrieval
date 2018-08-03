# -*- coding: utf-8 -*-

import functools
import tensorflow as tf
import hparams

CSV_COLUMN_NAMES["question", "answer", "lable"]

def create_vocab_table(vocab_file):
    return tf.contrib.lookup.index_table_from_file(vocab_file)

def parse_line(line, vocab_table):
    question, answer, lable = tf.decode_csv(line)
    question_ids = vocab_table.lookup(
            question.split(" ")[:hparams.max_question_len], tf.int64)
    answer_ids = vocab_table.lookup(answer.split(" ")[:hparams.max_answer_len], tf.int64)
    lable = int(lable)
    features = {} 
    features["question"] = question_ids
    features["question_len"] = len(question_ids)
    features["answer"] = answer_ids
    features["answer_len"] = len(answer_ids)
    features["lable"] = lable
    return features

def train_input_fn(filename, batch_size):
    dataset = tf.data.TextLineDataset(filename).skip(1)
    vocab_table = create_vocab_table(hparams.vocabulary_file)
    dataset = dataset.map(functools.partial(parse_line, vocab_table))
    dataset = dataset.shuffle(1000).repeat().batch_size(batch_size)
    return dataset

def create_feature_columns():
    feature_columns = []
    feature_columns.append(tf.feature_column.numeric_column(
        key="question", shape=hparams.max_question_len, dtype=tf.int64))
    feature_columns.append(tf.feature_column.numeric_column(
        key="question_len", dtype=tf.int64))
    feature_columns.append(tf.feature_column.numeric_column(
        key="answer", shape=hparams.max_answer_len, dtype=tf.int64))
    feature_columns.append(tf.feature_column.numeric_column(
        key="answer_len", dtype=tf.int64))
    feature_columns.append(tf.feature_column.numeric_column(
        key="lable", dtype=tf.int64))
    return feature_columns
