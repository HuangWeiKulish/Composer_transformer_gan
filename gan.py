# https://www.tensorflow.org/tutorials/text/transformer

import os
import glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pickle as pkl
from scipy import sparse

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, BatchNormalization, LayerNormalization, GaussianNoise, \
    Flatten, Reshape, Activation, GRU, RepeatVector, Dot, TimeDistributed, concatenate, \
    Bidirectional, Add, Permute, Dropout, Embedding, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

import util, transformer, convertor

# todo: check values after embedding: has both postive and negative!!!!


class LatentVector(tf.keras.Model):
    def __init__(self, embed_dim=256, nlayers=4, dim_base=4):
        # generate latent vector of dimension; (batch, in_seq_len, embed_dim)
        super(LatentVector, self).__init__()
        self.embed_dim = embed_dim
        self.nlayers = nlayers
        self.dim_list = [dim_base**i for i in range(1, nlayers+1)]

    def call(self, x_in):
        # x_in: (batch, in_seq_len, 1)
        x = x_in
        for dim in self.dim_list:
            x = Dense(dim, use_bias=True)(x)
        return x


class Discriminator(tf.keras.Model):

    def __init__(self, notes_pool_size, max_pos, embed_dim, dropout_rate=0.2, n_heads=4, fc_layers=3,
                 norm_epsilon=1e-6, fc_activation="relu", encoder_layers=3, decoder_layers=3, ):
        super(Discriminator, self).__init__()
        self.embedder = transformer.Embedder(notes_pool_size, max_pos, embed_dim, dropout_rate)
        self.notes_pool_size = notes_pool_size
        self.n_heads = n_heads
        self.depth = embed_dim // n_heads
        self.fc_layers = fc_layers
        self.norm_epsilon = norm_epsilon
        self.fc_activation = fc_activation
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

    def call(self, x_out_melody, x_out_duration, x_de_in, mask_padding, mask_lookahead):
        # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
        # x_out_duration: (batch, out_seq_len, 1)
        # x_de_in: (batch, 2): 2 columns: [tk.word_index['<start>'], 0]
        melody_ = util.softargmax(x_out_melody, beta=1e10)  # return the index of the max value: (batch, seq_len)
        melody_ = tf.expand_dims(melody_, axis=-1)  # (batch, seq_len, 1)
        music_ = tf.concat([melody_, x_out_duration], axis=-1)  # (batch, seq_len, 2): column: notes_id, duration
        emb = self.embedder(music_)  # (batch, seq_len, embed_dim)

        x_de_in = tf.expand_dims(x_de_in, 1)  # (batch, 1, 2)
        x_de = self.embedder(x_de_in)  # (batch, 1, embed_dim)

        # (batch, out_seq_len, 1)
        out, _ = transformer.TransformerBlocks.transformer(
            emb, x_de, mask_padding, mask_lookahead, out_notes_pool_size=self.notes_pool_size, embed_dim=self.embed_dim,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, n_heads=self.n_heads, depth=self.depth,
            fc_layers=self.fc_layers, norm_epsilon=self.norm_epsilon,
            transformer_dropout_rate=self.transformer_dropout_rate,
            fc_activation=self.fc_activation, type='')  # (None, None, 1)

        """
        out, _ = TransformerBlocks.transformer(
            emb, x_de, mask_padding, mask_lookahead, out_notes_pool_size=notes_pool_size, embed_dim=embed_dim,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, n_heads=n_heads, depth=depth,
            fc_layers=fc_layers, norm_epsilon=norm_epsilon,
            transformer_dropout_rate=transformer_dropout_rate,
            fc_activation=fc_activation, type='')
        """
        out = Dense(1, activation='sigmoid')(out)  # (None, None, 1)
        return out


class GAN:

    def __init__(self, g_in_seq_len=32,
                 g_de_max_pos=10000, embed_dim=256, n_heads=4,
                 g_out_notes_pool_size=8000, g_fc_activation="relu", g_encoder_layers=2, g_decoder_layers=2, g_fc_layers=3,
                 g_norm_epsilon=1e-6, g_transformer_dropout_rate=0.2, g_embedding_dropout_rate=0.2):









        # ---------------- generator encoder embedder --------------------
        self.g_encoder_embedder = Embedder(
            g_in_seq_len, g_out_notes_pool_size, g_en_max_pos, g_embed_dim, g_embedding_dropout_rate)





        self.g_input_dim = g_input_dim
        self.latent_mean = latent_mean
        self.latent_std = latent_std

        # generator model
        g_x_in = Input(shape=(g_input_dim, 1))
        g_x_out = Blocks.generator_block(
            g_x_in, fc_dim=g_fc_dim, fltr_dims=g_fltr_dims, knl_dims=g_knl_dims, up_dims=g_up_dims,
            bn_momentum=g_bn_momentum, noise_std=g_noise_std, final_fltr=g_final_fltr, final_knl=g_final_knl)
        self.gmodel = Model(inputs=[g_x_in], outputs=[g_x_out])
        if g_model_path_name is not None:
            self.gmodel = self.load_existing_model(self.gmodel, g_model_path_name, g_loaded_trainable, g_loss)
        if print_model:
            print(self.gmodel.summary())

        # discriminator model
        d_x_in = Input(shape=(self.gmodel.output.shape[1], 1))
        d_x_out = Blocks.discriminator_block(
            d_x_in, fltr_dims=d_fltr_dims, knl_dims=d_knl_dims, down_dims=d_down_dims, bn_momentum=d_bn_momentum)
        self.dmodel = Model(inputs=[d_x_in], outputs=[d_x_out])
        if d_model_path_name is not None:
            self.dmodel = self.load_existing_model(self.dmodel, d_model_path_name, d_loaded_trainable)
        if print_model:
            print(self.dmodel.summary())
        self.dmodel.compile(loss=d_loss, optimizer=Adam(lr=d_lr, clipnorm=d_clipnorm), metrics=d_metrics)

        # full model
        self.dmodel.trainable = False  # freeze discriminator while training the full model
        gan_out = self.dmodel(g_x_out)
        self.full_model = Model(inputs=[g_x_in], outputs=[gan_out])
        if print_model:
            print(self.full_model.summary())
        self.full_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=f_lr, clipnorm=f_clipnorm))









    @staticmethod
    def true_samples(all_true_x, n_samples=20):
        inds = np.random.randint(0, all_true_x.shape[0], n_samples)
        true_x = all_true_x[inds]
        return true_x

    def fake_samples(self, n_samples=20):
        x_in = DataPreparation.latant_vector(n_samples, self.g_input_dim, mean_=self.latent_mean, std_=self.latent_std)
        return self.gmodel.predict(x_in)

    def load_existing_model(self, model, load_model_path_name, loaded_trainable, loss_func):
        updated_model = load_model(load_model_path_name) if loss_func is None \
            else load_model(load_model_path_name, custom_objects={'loss_func': loss_func})
        for i in range(len(updated_model.layers)):
            model.layers[i] = updated_model.layers[i]
            model.layers[i].trainable = loaded_trainable
        return model

    def train(self, all_true_x, n_epoch=100, n_samples=20, save_step=10, n_save=1, verbose=True, save_pic=True,
              file_save_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/result',
              model_save_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/models'):
        true_y, fake_y = np.ones((n_samples, 1), dtype=np.float32), np.zeros((n_samples, 1), dtype=np.float32)
        self.d_perform, self.f_perform = [], []
        for i in range(n_epoch):
            true_x = GAN.true_samples(all_true_x, n_samples=n_samples)
            fake_x = self.fake_samples(n_samples=n_samples)
            combine_x = np.array(fake_x.tolist()+true_x.tolist())
            combine_y = np.array(fake_y.tolist()+true_y.tolist())

            # train discriminator on combined samples
            d_loss, d_metr = self.dmodel.train_on_batch(combine_x, combine_y)
            self.d_perform.append([d_loss, d_metr])

            # generate latent vector input
            latent_x = DataPreparation.latant_vector(
                n_samples*2, self.g_input_dim, mean_=self.latent_mean, std_=self.latent_std)
            f_loss = self.full_model.train_on_batch(latent_x, np.ones((n_samples*2, 1), dtype=np.float32))  # inverse label!!!!
            # print result
            if verbose:
                print('epoch {}: d_loss={}, d_metr={}, f_loss={}'.format(i+1, d_loss, d_metr, f_loss))

            # save predictions
            if (i > 0) & (i % save_step == 0):
                print('save result')
                self.gmodel.save(os.path.join(model_save_path, 'gmodel.h5'))
                self.dmodel.save(os.path.join(model_save_path, 'dmodel.h5'))
                fake_samples_save = fake_x[:n_save]
                for j, f_s in enumerate(fake_samples_save):
                    file_name = '{}_{}'.format(i, j)
                    f_s = DataPreparation.recover_array(f_s)
                    pkl.dump(sparse.csr_matrix(f_s), open(os.path.join(file_save_path, file_name+'.pkl'), 'wb'))
                    if save_pic:
                        plt.figure(figsize=(20, 5))
                        plt.plot(range(f_s.shape[0]), np.multiply(f_s, range(1, 89)), marker='.',
                                 markersize=1, linestyle='')
                        plt.savefig(os.path.join(file_save_path, file_name + '.png'))
                        plt.close()


#======================================

import tensorflow_datasets as tfds

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)


sample_string = 'Transformer is awesome.'
tokenized_string = tokenizer_en.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))
original_string = tokenizer_en.decode(tokenized_string)
print('The original string: {}'.format(original_string))

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))


BUFFER_SIZE = 20000
BATCH_SIZE = 64


def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size + 1]
    return lang1, lang2


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en


MAX_LENGTH = 40


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

pt_batch, en_batch = next(iter(val_dataset))

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

# --------------------------------
encoder_layers, decoder_layers = 2, 2
embed_dim = 512
n_heads = 8
fc_layers = 2048
in_notes_pool_size = 8500
out_notes_pool_size = 8000,
en_max_pos = 10000
de_max_pos = 6000


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(embed_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
temp_learning_rate_schedule = CustomSchedule(embed_dim)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# for (batch, (inp, tar)) in enumerate(train_dataset):
#     print(train_dataset)


tmp = list(train_dataset)[0:10]
inp, tar = tmp[-1]





tar_inp = tar[-1]
tar_real = tar[1:]


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp[0], tar[0])







num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)



"""
filepath = '/Users/Wei/Desktop/piano_classic/Chopin_array/nocturne_c_sharp-_(c)unknown.pkl'

in_seq_len=128
out_seq_len=512
en_max_pos=5000
de_max_pos=50000
embed_dim=256
n_heads=4
depth=2
in_notes_pool_size=8000
out_notes_pool_size=8000
embedding_dropout_rate=0.2
fc_activation="relu"
"""

"""
g_input_dim=100, g_fc_dim=100, g_fltr_dims=(20, 10, 5, 3, 1, 1), g_knl_dims=(5, 7, 9, 11, 13, 15),
                 g_up_dims=(2, 2, 2, 2, 2, 2), g_bn_momentum=0.6, g_noise_std=0.2, g_final_fltr=1, g_final_knl=5,
                 g_model_path_name=None, g_loaded_trainable=False, g_loss=None,
                 d_fltr_dims=(1, 1, 3, 5, 10, 20), d_knl_dims=(15, 13, 11, 9, 7, 5),
                 d_down_dims=(2, 2, 2, 2, 2, 2), d_bn_momentum=0.6, d_model_path_name=None, d_loaded_trainable=False,
                 d_lr=0.0001, d_clipnorm=1.0, d_loss='mse', d_metrics=['accuracy'],
                 f_lr=0.01, f_clipnorm=1.0, print_model=True, latent_mean=0.0, latent_std=1.1


"""