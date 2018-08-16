#-*- coding: utf-8 -*-

import functools
import csv
import tensorflow as tf
import numpy as np

CSV_COLUMN_NAMES = ["question", "answer", "lable"]

# padding all string seqence to this length, make them easier to process
TEXT_FEATURE_SIZE = 160

def create_vocab_table(vocab_file):
    """unknown word will return 0"""
    return tf.contrib.lookup.index_table_from_file(vocab_file, default_value=0)

def parse_line(line):
    """parse csv format line to fields"""
    fields = tf.decode_csv(line, record_defaults=[[""], [""], [0]])
    return fields

def padding_string_sequence(question, question_len, answer, answer_len, lable):
    question_padding = [[0, TEXT_FEATURE_SIZE - tf.shape(question)[0]]]
    answer_padding = [[0, TEXT_FEATURE_SIZE - tf.shape(answer)[0]]]
    question = tf.pad(question, question_padding)
    answer = tf.pad(answer, answer_padding)
    return question, question_len, answer, answer_len, lable

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

    # padding question and answer
    dataset = dataset.map(padding_string_sequence)
    
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

def parse_test_data_line(line, vocab_table, hparams):
    fields = tf.decode_csv(line,
            record_defaults=[[""],[""],[""],[""],[""],[""],[""],[""],[""],[""],[""]])

    # convert to features dict
    features = {}
    question, answer = fields[:2]
    features["question"] = tf.string_split([question]).values
    features["question"] = vocab_table.lookup(features["question"])
    features["question"] = features["question"][:hparams.max_question_len]
    features["question_len"] = tf.size(features["question"])
    features["answer"] = tf.string_split([answer]).values
    features["answer"] = vocab_table.lookup(features["answer"])
    features["answer"] = features["answer"][:hparams.max_answer_len]
    features["answer_len"] = tf.size(features["answer"])

    paddings = [[0, TEXT_FEATURE_SIZE - features["question_len"]]]
    features["question"] = tf.pad(features["question"], paddings)
    paddings = [[0, TEXT_FEATURE_SIZE - features["answer_len"]]]
    features["answer"] = tf.pad(features["answer"], paddings)

    distractors = fields[2:]
    for i, distractor in enumerate(distractors):
        split_distractor = tf.string_split([distractor]).values
        key = "distractor_{}".format(i)
        features[key] = vocab_table.lookup(split_distractor)
        features[key] = features[key][:hparams.max_answer_len]
        len_key = "{}_len".format(key)
        features[len_key] = tf.size(features[key])
        paddings = [[0, TEXT_FEATURE_SIZE - features[len_key]]]
        features[key] = tf.pad(features[key], paddings)
    
    # In evaluation we have 10 classes
    # The first one (index 0) is always the correct one
    lable = tf.constant(0, dtype=tf.int64)
    return features, lable

def test_input_fn(hparams):
    #read test file and skip header line
    dataset = tf.data.TextLineDataset(hparams.test_file).skip(1)

    """
    #parse line
    dataset = dataset.map(
            lambda line: tf.decode_csv(line,
                record_defaults=[[""],[""],[""],[""],[""],[""],[""],[""],[""],[""],[""]]))
    
    # string split
    dataset = dataset.map(
            lambda fields: [tf.string_split([field]).values() for field in fields])
    """
    vocab_table = create_vocab_table(hparams.vocabulary_file)
    dataset = dataset.map(functools.partial(parse_test_data_line,
                            vocab_table=vocab_table, hparams=hparams))

    dataset = dataset.padded_batch(
            hparams.batch_size,
            padded_shapes=(
                {
                    "question":tf.TensorShape([None]),
                    "question_len":tf.TensorShape([]),
                    "answer":tf.TensorShape([None]),
                    "answer_len":tf.TensorShape([]),
                    "distractor_0":tf.TensorShape([None]),
                    "distractor_0_len":tf.TensorShape([]),
		    "distractor_1":tf.TensorShape([None]),
                    "distractor_1_len":tf.TensorShape([]),
                    "distractor_2":tf.TensorShape([None]),
                    "distractor_2_len":tf.TensorShape([]),
                    "distractor_3":tf.TensorShape([None]),
                    "distractor_3_len":tf.TensorShape([]),
                    "distractor_4":tf.TensorShape([None]),
                    "distractor_4_len":tf.TensorShape([]),
                    "distractor_5":tf.TensorShape([None]),
                    "distractor_5_len":tf.TensorShape([]),
                    "distractor_6":tf.TensorShape([None]),
                    "distractor_6_len":tf.TensorShape([]),
                    "distractor_7":tf.TensorShape([None]),
                    "distractor_7_len":tf.TensorShape([]),
                    "distractor_8":tf.TensorShape([None]),
                    "distractor_8_len":tf.TensorShape([])
                },
                tf.TensorShape([])))

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
