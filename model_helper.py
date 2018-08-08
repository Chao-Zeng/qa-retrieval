#-*- coding: utf-8 -*-

import functools
import csv
import tensorflow as tf
import numpy as np

CSV_COLUMN_NAMES = ["question", "answer", "lable"]

def create_vocab_table(vocab_file):
    """unknown word will return 0"""
    return tf.contrib.lookup.index_table_from_file(vocab_file, default_value=0)

def parse_line(line):
    """parse csv format line to fields"""
    fields = tf.decode_csv(line, record_defaults=[[""], [""], [0]])
    return fields

def train_input_fn(hparams):
    # create Dataset by train file and skip header line
    dataset = tf.data.TextLineDataset(hparams.train_file).skip(1)
    # parse csv
    dataset = dataset.map(parse_line)
    # split string
    dataset = dataset.map(
            lambda question, answer, lable:(
                tf.string_split([question]).values,
                tf.string_split([answer]).values,
                lable))
    # filter question and answer length
    dataset = dataset.filter(
            lambda question, answer, lable:(
                tf.logical_and(tf.size(question) > 0, tf.size(answer) > 0)))
    dataset = dataset.map(
            lambda question, answer, lable:(
                question[:hparams.max_question_len],
                answer[:hparams.max_answer_len],
                lable))
    # convert word strings to ids
    vocab_table = create_vocab_table(hparams.vocabulary_file)
    dataset = dataset.map(
            lambda question, answer, lable:(
                vocab_table.lookup(question),
                vocab_table.lookup(answer),
                lable))
    
    # add in question and answer sequence length
    dataset = dataset.map(
            lambda question, answer, lable:(
                question, tf.size(question), answer, tf.size(answer), lable))
    
    # convert to features dict and lable tuple
    dataset = dataset.map(
            lambda question, question_len, answer, answer_len, lable:(
                {
                    "question":question,
                    "question_len":question_len,
                    "answer":answer,
                    "answer_len":answer_len
                 },
                lable))

    # shuffle and repeat
    dataset = dataset.shuffle(1000).repeat()

    # padded batch as question and answer have varying size
    dataset = dataset.padded_batch(
            hparams.batch_size,
            padded_shapes=(
                {
                    "question":tf.TensorShape([None]),
                    "question_len":tf.TensorShape([]),
                    "answer":tf.TensorShape([None]),
                    "answer_len":tf.TensorShape([])
                },
                tf.TensorShape([])))
    
    # create features and lable
    #iterator = dataset.make_initializable_iterator()
    #tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    """
    question, question_len, answer, answer_len, lable = iterator.get_next()
    features = dict()
    features["question"] = question
    features["question_len"] = question_len
    features["answer"] = answer
    features["answer_len"] = answer_len
    """

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

def load_vocab(vocab_file):
    """load vocabulary file and return a words list and word count"""
    vocab = []
    vocab_size = 0
    with open(vocab_file) as f:
        for word in f:
            vocab.append(word.strip())
            vocab_size += 1
    return vocab, vocab_size

def load_glove(embed_file):
    """Load embed_file into a python dictionary.
    Args:
        embed_file: file path to the embedding file.
    Returns:
        a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    embed_dict = dict()
    embed_size = None
    with open(embed_file, 'r', encoding = 'utf-8') as f:
        for line in f:
            tokens = line.strip().split(" ")
            word  = tokens[0]
            word_vec = list(map(float, tokens[1:]))
            embed_dict[word] = word_vec
            if embed_size:
                assert embed_size == len(word_vec), "All embedding size should be same."
            else:
                embed_size = len(word_vec)

    return embed_dict, embed_size

def create_pretrained_embedding(vocab_file, embed_file):
    """Load pretrain embeding from embed_file, and return an embedding matrix.
    Args:
        vocab_file: vocabulary file
        embed_file: Glove embedding file
    """
    vocab, vocab_size = load_vocab(vocab_file)
    embed_dict, embed_size = load_glove(embed_file)
    default_embed_vec = [0.0] * embed_size
    embed_matrix = np.array(
            [embed_dict.get(word, default_embed_vec) for word in vocab],
            dtype=np.float32)

    #embed_matrix = tf.constant(embed_matrix)
    embedding = tf.get_variable("word_embeddings",
            initializer=embed_matrix)

    return embedding

def get_embeddings(hparams):
    if hparams.glove_file and hparams.vocabulary_file:
        tf.logging.info("Loading Glove embeddings...")
        embedding = create_pretrained_embedding(
                hparams.vocabulary_file, hparams.glove_file)
    else:
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initial_embeddings = tf.random_uniform_initializer(-0.25, 0.25)
        vocab_size = hparams.vocab_size
        embedding_size = hparams.embedding_size
        embedding = tf.get_variable("word_embeddings",
                shape=[vocab_size, embedding_size],
                initializer=initial_embeddings)
    
    return embedding
