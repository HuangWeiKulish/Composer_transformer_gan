import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def latant_vector(n_samples, input_dim, mean_=0.0, std_=1.1):
    return np.random.normal(mean_, std_, size=[n_samples, input_dim, 2])


def load_existing_model(model, load_model_path_name, load_trainable, loss_func):
    updated_model = load_model(load_model_path_name) if loss_func is None \
        else load_model(load_model_path_name, custom_objects={'loss_func': loss_func})
    for i in range(len(updated_model.layers)):
        model.layers[i] = updated_model.layers[i]
        model.layers[i].trainable = load_trainable
    return model


def loss_func(real, pred_melody, pred_duration, cross_entropy):
    loss_ = cross_entropy(real, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def padding_mask(x_):
    return tf.cast(tf.math.equal(x_, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]


def lookahead_mask(seq_len):
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)


def checkpoint(model, save_path, optimizer, save_every=20):
    cp = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
    cp_manager = tf.train.CheckpointManager(cp, save_path, max_to_keep=save_every)
    if cp_manager.latest_checkpoint:
        cp.restore(cp_manager.latest_checkpoint)  # restore the latest checkpoint if exists


def load_true_data():
    pass



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, embed_dim, warmup_steps=4000):
        # https://www.tensorflow.org/tutorials/text/transformer
        super(CustomSchedule, self).__init__()

        self.embed_dim = embed_dim
        self.embed_dim = tf.cast(self.embed_dim, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embed_dim) * tf.math.minimum(arg1, arg2)








