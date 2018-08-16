import tensorflow as tf
import model
import model_helper

def evaluate(hparams):
    estimator = tf.estimator.Estimator(
            model_fn = model.model_fn,
            model_dir=hparams.model_dir,
            params={
                "hparams":hparams
                })

    estimator.evaluate(
            input_fn=lambda:model_helper.test_input_fn(hparams),
            steps=None)
