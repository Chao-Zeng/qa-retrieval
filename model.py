import tensorflow as tf
import dual_encoder
import model_helper

def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return dual_encoder.dual_encoder_model(
                features, labels, mode, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return dual_encoder.dual_encoder_model(
                features, labels, mode, params)

    if mode == tf.estimator.ModeKeys.EVAL:
        # We have 10 exampels per record, so we accumulate them
        all_questions = [features["question"]]
        all_question_lens = [features["question_len"]]
        all_answers = [features["answer"]]
        all_answer_lens = [features["answer_len"]]

        batch_size = tf.shape(features["question"])[0]
        # real answer's label is 1
        all_labels = [tf.ones([batch_size], dtype=tf.int64)]

        for i in range(9):
            distractor_key = "distractor_{}".format(i)
            distractor_len_key = "distractor_{}_len".format(i)
            all_questions.append(features["question"])
            all_question_lens.append(features["question_len"])
            all_answers.append(features[distractor_key])
            all_answer_lens.append(features[distractor_len_key])
            # distractor's label always 0
            all_labels.append(tf.zeros([batch_size], dtype=tf.int64))

        all_features = {}
        all_features["question"] = tf.concat(all_questions, 0)
        all_features["question_len"] = tf.concat(all_question_lens, 0)
        all_features["answer"] = tf.concat(all_answers, 0)
        all_features["answer_len"] = tf.concat(all_answer_lens, 0)
        all_labels = tf.concat(all_labels, 0)

        eval_result = dual_encoder.dual_encoder_model(
                all_features, all_labels, mode, params)

        # get probabilities for each answer(distractor) in one example
        split_probs = tf.split(eval_result.predictions, 10, axis=0)
        class_probs = tf.concat(split_probs, 1)

        # Because the true response is always element 0 in array,
        # the label for each example is 0.
        metrics = {}
        for k in [1,2,5,10]:
            metrics["recall_at_{}".format(k)] = tf.contrib.metrics.streaming_sparse_recall_at_k(
                        class_probs, labels, k)

        return tf.estimator.EstimatorSpec(
                mode=mode, loss=eval_result.loss, eval_metric_ops=metrics)
