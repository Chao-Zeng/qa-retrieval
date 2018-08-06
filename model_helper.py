# -*- coding: utf-8 -*-

import functools
import tensorflow as tf
import numpy as np
import hparams

CSV_COLUMN_NAMES["question", "answer", "lable"]

"""
def create_vocab_table(vocab_file):
    return tf.contrib.lookup.index_table_from_file(vocab_file)
"""
def load_vocab(filename):
    vocab = None
    with open(filename) as f:
        vocab = f.read().splitlines()
    dct = {}
    for index, word in enumerate(vocab):
        dct[word] = index
    return dct

def load_glove(filename):
    dct = {}
    index = 0
    with open(filename, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            tokens = line.split(" ")
            word  = tokens[0]
            word_vec = tokens[1:]
            dct[word] = word_vec

    return dct

def parse_line(line, vocab_dict):
    question, answer, lable = tf.decode_csv(line)
    question_ids = [vocab_dict[word] for word in question.split(" ")[:hparams.max_question_len]]
    answer_ids = [vocab_dict[word] for word in answer.split(" ")[:hparams.max_answer_len]]
    lable = int(lable)
    features = {} 
    features["question"] = question_ids
    features["question_len"] = len(question_ids)
    features["answer"] = answer_ids
    features["answer_len"] = len(answer_ids)
    #features["lable"] = lable
    return features, lable

def train_input_fn(filename, batch_size):
    dataset = tf.data.TextLineDataset(filename).skip(1)
    vocab_dict = load_vocab(hparams.vocabulary_file)
    dataset = dataset.map(functools.partial(parse_line, vocab_dict))
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
    return feature_columns

def get_embeddings(hparams):
    if hparams.glove_file and hparams.vocabulary_file:
        tf.logging.info("Loading Glove embeddings...")
        vocab_dict = load_vocab(hparams.vocabulary_file)
        glove_dict = load_glove(hparams.glove_file)
        vocab_size = len(vocab_dict.items())
        embedding_dim = len(glove_dict.values()[0])
        initial_embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_dict), embedding_dim)).astype("float32")
        for word, word_id in vocab_dict.items():
            initial_embeddings[word_id, :] = glove_dict[word]
    else:
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initial_embeddings = tf.random_uniform_initializer(-0.25, 0.25)
        vocab_size = hparams.vocab_size
        embedding_dim = hparams.embedding_dim
    
    return tf.get_variable("word_embeddings", shape=[vocab_size, embedding_dim], initializer=initial_embeddings)
