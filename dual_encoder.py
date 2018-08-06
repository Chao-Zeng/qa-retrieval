import tensorflow as tf
import model_helper
import hparams

def dual_encoder_model(features, labels, mode, params):
    embeddings_W = model_helper.get_embeddings(hparams)
    question_embedded = tf.nn.embedding_lookup(embeddings_W,
            features["question"], name="question_embedded")
    answer_embedded = tf.nn.embedding_lookup(embeddings_W,
            features["answer"], nmae="answer_embedded")

    cell = tf.contrib.rnn.LSTMCell(num_units=hparams.rnn_num_units,
                                   use_peepholes=True)

    question_len = features["question_len"]
    answer_len = features["answer_len"]
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell,
            tf.concat([question_embedded, answer_embedded], 0),
            sequence_length=tf.concat([question_len, answer_len], 0),
            dtype = tf.float32)
    encoding_question, encoding_answer = tf.split(rnn_states.h, 2, 0)

    M = tf.get_variable(
            name="M",
            shape=[hparams.rnn_num_units, hparams.rnn_num_units],
            initializer=tf.truncated_normal_initializer())

    generated_response = tf.matmul(encoding_question, M)
    generated_response = tf.expand_dims(generated_response, 2)
    encoding_answer = tf.expand_dims(encoding_answer, 2)

    logits = tf.matmul(generated_response, encoding_answer, True)
    logits = tf.squeeze(logits, [2])

    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)

    if mode == tf.contrib.learn.ModeKeys.INFER:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=probs)

    losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=tf.to_float(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")

    # train_op
    optimizer = tf.train.AdagradOptimizer(learning_rate=hparams.learning_rate)
    train_op =  optimizer.minimize(mean_loss,
                        global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=mean_loss,
            train_op=train_op)

