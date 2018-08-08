import tensorflow as tf
import dual_encoder
import model_helper

def train(hparams):
    estimator = tf.estimator.Estimator(
            model_fn = dual_encoder.dual_encoder_model,
            model_dir=hparams.model_dir,
            params={
                "hparams":hparams
                })

    estimator.train(
            input_fn=lambda:model_helper.train_input_fn(hparams),
            steps=hparams.train_steps)
