import tensorflow as tf
import model
import model_helper

def train(hparams):
    estimator = tf.estimator.Estimator(
            model_fn = model.model_fn,
            model_dir=hparams.model_dir,
            params={
                "hparams":hparams
                })

    estimator.train(
            input_fn=lambda:model_helper.train_input_fn(hparams),
            steps=hparams.train_steps)
