import tensorflow as tf
import model_helper

def dual_encoder_model(features, labels, mode, params):
    hparams = params["hparams"]
    embeddings = model_helper.get_embeddings(hparams)

    # embed question and answer
    question_embedded = tf.nn.embedding_lookup(embeddings,
            features["question"], name="question_embedded")
    answer_embedded = tf.nn.embedding_lookup(embeddings,
            features["answer"], name="answer_embedded")

    question_len = features["question_len"]
    answer_len = features["answer_len"]

    # build a LSTM Cell
    # if cell is LSTM Cell, state is LSTMStateTuple (c, h),
    # c is hidden state and h is output
    """
    with tf.variable_scope("question_rnn") as question_scope:
        cell_q = tf.contrib.rnn.BasicLSTMCell(num_units=hparams.rnn_num_units)
        question_rnn_output, question_rnn_states = tf.nn.dynamic_rnn(
                                                cell_q,
                                                question_embedded,
                                                sequence_length=question_len,
                                                dtype=tf.float32,
                                                scope=question_scope)
        encoding_question = question_rnn_states.h

    with tf.variable_scope("answer_rnn") as answer_scope:
        cell_a = tf.contrib.rnn.BasicLSTMCell(num_units=hparams.rnn_num_units)
        answer_rnn_output, answer_rnn_states = tf.nn.dynamic_rnn(
                                                cell_a,
                                                answer_embedded,
                                                sequence_length=answer_len,
                                                #initial_state=question_rnn_states,
                                                dtype=tf.float32,
                                                scope=answer_scope)
        encoding_answer = answer_rnn_states.h
    """
    with tf.variable_scope("rnn") as rnn_scope:
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hparams.rnn_num_units)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell,
                tf.concat([question_embedded, answer_embedded], 0),
                sequence_length = tf.concat([question_len, answer_len], 0),
                dtype = tf.float32)

        encoding_question, encoding_answer = tf.split(rnn_states.h, 2, 0)


    M = tf.get_variable(
            name="M",
            shape=[hparams.rnn_num_units, hparams.rnn_num_units],
            initializer=tf.truncated_normal_initializer())

    # "Predict" a  response: question * M
    generated_response = tf.matmul(encoding_question, M)
    # add one dims to generated_response,
    # (batch_size, embed_size) -> (batch_size, embed_size, 1)
    generated_response = tf.expand_dims(generated_response, 2)

    # add one dims to encoding_answer,
    # (batch_size, embed_size) -> (batch_size, embed_size, 1)
    encoding_answer = tf.expand_dims(encoding_answer, 2)

    # (batch_size,) -> (batch_size, 1)
    labels = tf.expand_dims(labels, 1)

    # Dot product between generated response and actual response
    # transport(generated_response) * encoding_answer
    # logits shape is (batch_size, 1, 1)
    logits = tf.matmul(generated_response, encoding_answer, True)
    logits = tf.squeeze(logits, [2])

    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=probs)

    # cross entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=tf.to_float(labels))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                predictions=probs, loss=mean_loss)

    # train mode
    assert mode == tf.estimator.ModeKeys.TRAIN

    # train_op
    optimizer = tf.train.AdagradOptimizer(learning_rate=hparams.learning_rate)
    train_op =  optimizer.minimize(mean_loss,
                        global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=mean_loss,
            train_op=train_op)

