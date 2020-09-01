import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import preprocess

tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def latant_vector(n_samples, input_dim, mean_=0.0, std_=1.1):
    return np.random.normal(mean_, std_, size=[n_samples, input_dim, 2])


def padding_mask(x_):
    # pad position (pad with 0) = 1, non-pad position = 0
    # return shape: (batch, 1, 1, seq_len)
    return tf.cast(tf.math.equal(x_, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]


def lookahead_mask(seq_len):
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)


def checkpoint(model, save_path, optimizer, save_every=20):
    cp = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
    cp_manager = tf.train.CheckpointManager(cp, save_path, max_to_keep=save_every)
    if cp_manager.latest_checkpoint:
        cp.restore(cp_manager.latest_checkpoint)  # restore the latest checkpoint if exists


def softargmax(x, beta=1e10):
    x = tf.convert_to_tensor(x)
    x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
    return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)


def number_encode_text(x, tk):
    # x: (batch, length, 4); 4 columns: [notes in text format, velocity, time since last start, notes duration]
    # ------------- encode notes from text to integer --------------
    x_0 = tk.texts_to_sequences(x[:, :, 0].tolist())
    return np.append(np.expand_dims(x_0, axis=-1), x[:, :, 1:], axis=-1).astype(np.float32)


def load_true_data(tk, in_seq_len, out_seq_len, step=60, batch_size=50,
                   pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['']):
    # tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_dict_final.pkl'
    # tk = pkl.load(open(tk_path, 'rb'))
    x_in, x_tar = preprocess.DataPreparation.batch_preprocessing(
        in_seq_len, out_seq_len-1, step, pths, name_substr_list)
    batch = x_in.shape[0]

    # append '<start>' in front and '<end>' at the end
    x_tar = np.concatenate(
        [np.expand_dims(np.array([['<start>', 0, 0, 0]] * batch, dtype=object), axis=1),
         x_tar,
         np.expand_dims(np.array([['<end>', 0, 0, 0]] * batch, dtype=object), axis=1)], axis=1)

    x_in_ = number_encode_text(x_in, tk)
    x_tar_ = number_encode_text(x_tar, tk)
    dataset = tf.data.Dataset.from_tensor_slices((x_in_, x_tar_[:, :-1, :], x_tar_[:, 1:, :])).cache()
    dataset = dataset.shuffle(x_in.shape[0]+1).batch(batch_size)
    return tk, dataset


def inds2notes(tk, nid, default='p'):
    try:
        return tk.index_word[nid]
    except:
        return default


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
def loss_func_notes(real, pred):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=tf.float32)
    loss_ = cross_entropy(real, pred)
    loss_ *= mask  # (batch, out_seq_len)
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)  # scalar


@tf.function
def loss_func_duration(real, pred):
    loss_ = tf.keras.losses.MSE(real, pred)
    return tf.reduce_sum(loss_) / (real.shape[0] * real.shape[1])



