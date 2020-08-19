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
import util
import transformer
import convertor
import discriminator
import generator


class LatentVector(tf.keras.Model):

    def __init__(self, embed_dim=256, nlayers=4, dim_base=4):
        # generate latent vector of dimension; (batch, in_seq_len, embed_dim)
        super(LatentVector, self).__init__()
        #self.trainable = True
        self.embed_dim = embed_dim
        self.nlayers = nlayers
        self.dim_list = [dim_base**i for i in range(1, nlayers+1)]

    def call(self, x_in):
        # x_in: (batch, in_seq_len, 1)
        x = x_in
        for dim in self.dim_list:
            x = Dense(dim, use_bias=True)(x)
        return x


class GAN:

    def __init__(self, g_in_seq_len=16, latent_nlayers=4, latent_dim_base=4,
                 g_out_seq_len=64,
                 g_de_max_pos=10000, embed_dim=256, n_heads=4,
                 g_out_notes_pool_size=8000, g_fc_activation="relu", g_encoder_layers=2, g_decoder_layers=2,
                 g_fc_layers=3,
                 g_norm_epsilon=1e-6, g_transformer_dropout_rate=0.2, g_embedding_dropout_rate=0.2, beta=1e10,
                 loss='mean_squared_error', metrics=['mae'], opt=Adam(lr=0.1, clipnorm=1.0),
                 ):

        # g_de_init: tf.expand_dims(tf.constant([[tk.word_index['<start>'], 0]] * batch_size, dtype=tf.float32), 1)
        g_de_init = Input((1, 2))
        latent_in = Input((g_in_seq_len, 1))

        # ========================== Generator ==========================
        # generate latent vector -------------------------
        # g_in: (batch, g_in_seq_len, embed_dim)
        g_in = LatentVector(embed_dim=embed_dim, nlayers=latent_nlayers, dim_base=latent_dim_base)(latent_in)

        # generate latent vector -------------------------
        gen = generator.Generator(
            de_max_pos=g_de_max_pos, embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=g_out_notes_pool_size,
            fc_activation=g_fc_activation, encoder_layers=g_encoder_layers, decoder_layers=g_decoder_layers,
            fc_layers=g_fc_layers, norm_epsilon=g_norm_epsilon, transformer_dropout_rate=g_transformer_dropout_rate,
            embedding_dropout_rate=g_embedding_dropout_rate)
        g_out = GAN.generate_music(g_in, g_de_init, g_out_seq_len, gen, beta=beta)
        self.gmodel = Model(inputs=[latent_in, g_de_init], outputs=g_out)
        # Todo: gmodel.summary() shows Total params: 0, Trainable params: 0, Non-trainable params: 0

        # tmp = Model(inputs=latent_in, outputs=g_in)
        # tmp.summary()


        # gmodel.compile(loss=loss, optimizer=opt, metrics=metrics)

        # ========================== Discriminator ==========================





        #x_en_melody, x_de_in, mask_padding, mask_lookahead



        # ---------------- generator encoder embedder --------------------
        #self.g_encoder_embedder =





        # self.g_input_dim = g_input_dim
        # self.latent_mean = latent_mean
        # self.latent_std = latent_std

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
    def generate_music(g_in, g_de_init, g_out_seq_len, gen, beta=1e10):
        # gen is the generator model
        # g_in: (batch, g_in_seq_len, embed_dim)
        # g_de_init: (batch, 1, 2):
        #       tf.expand_dims(tf.constant([[tk.word_index['<start>'], 0]] * batch_size, dtype=tf.float32), 1)
        de_in = g_de_init
        for i in range(g_out_seq_len):
            # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
            # x_out_duration: (batch, out_seq_len, 1)
            x_out_melody, all_weights_melody, x_out_duration, all_weights_duration = gen(
                g_in, de_in, mask_padding=None, mask_lookahead=None)
            melody_ = util.softargmax(x_out_melody, beta=beta)
            melody_ = tf.expand_dims(melody_, axis=-1)  # (None, None, 1)
            music_ = tf.concat([melody_, x_out_duration], axis=-1)  # (None, None, 2): column: notes_id, duration
            de_in = tf.concat((de_in, music_), axis=1)
        return de_in














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


"""
filepath = '/Users/Wei/Desktop/piano_classic/Chopin_array/nocturne_c_sharp-_(c)unknown.pkl'

g_in_seq_len=16
latent_nlayers=4
latent_dim_base=4
g_out_seq_len=64
g_de_max_pos=10000
embed_dim=256
n_heads=4
g_out_notes_pool_size=8000
g_fc_activation="relu"
g_encoder_layers=2
g_decoder_layers=2
g_fc_layers=3
g_norm_epsilon=1e-6
g_transformer_dropout_rate=0.2
g_embedding_dropout_rate=0.2
beta=1e10
"""
