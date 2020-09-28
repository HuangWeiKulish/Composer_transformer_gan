import numpy as np
import itertools
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import preprocess

tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def padding_mask(x_):
    # pad position (pad with 0) = 1, non-pad position = 0
    # return shape: (batch, 1, 1, seq_len)
    return tf.cast(tf.math.equal(x_, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]


def lookahead_mask(seq_len):
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)


# def softargmax(x, beta=1e10):
#     x = tf.convert_to_tensor(x)
#     x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
#     return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)
#
#
# def number_encode_text(x, tk, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3):
#     # x: (batch, length, 4); 4 columns: [chords in text format, velocity, time since last start, chords duration]
#     # ------------- encode chords from text to integer --------------
#     x_0 = tk.texts_to_sequences(x[:, :, 0].tolist())
#     return np.append(np.expand_dims(x_0, axis=-1) - 1,  # ..- 1 because tk index starts from 1
#                      np.divide(x[:, :, 1:], np.array([vel_norm, tmps_norm, dur_norm])), axis=-1).astype(np.float32)


def load_true_data_pretrain_gen(
        in_seq_len, out_seq_len, step=60, batch_size=50,
        pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['']):
    x_in, x_tar = preprocess.DataPreparation.batch_preprocessing_pretrain_gen(
        in_seq_len, out_seq_len, step, pths, name_substr_list)
    dataset = tf.data.Dataset.from_tensor_slices((x_in.astype(str), x_tar.astype(str))).cache()
    dataset = dataset.shuffle(x_in.shape[0]+1).batch(batch_size)
    return dataset


def load_true_data_gan(
        seq_len, step=60, batch_size=50,
        pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''], remove_same_chords=False):
    x_in = preprocess.DataPreparation.batch_preprocessing_gan(
        seq_len, step=step, pths=pths, name_substr_list=name_substr_list)  # (batch, seq_len, 4)
    if remove_same_chords:
        x_in = [ary for ary in x_in if len(np.unique(ary[:, 0])) > 1]
        x_in = np.stack(x_in)
    dataset = tf.data.Dataset.from_tensor_slices(x_in.astype(str)).cache()
    dataset = dataset.shuffle(x_in.shape[0]+1).batch(batch_size)
    return dataset


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


@tf.function
def loss_func_chords(real, pred):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=tf.float32)
    loss_ = cross_entropy(real, pred)
    loss_ *= mask  # (batch, out_seq_len)
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)  # scalar


@tf.function
def loss_func_time(real, pred):
    loss_ = tf.keras.losses.MSE(real, pred)
    return tf.reduce_sum(loss_) / (real.shape[0] * real.shape[1])


def model_trainable(model, trainable=True):
    try:
        model.trainable = trainable
    except:
        for layer in model.layers:
            layer.trainable = trainable

