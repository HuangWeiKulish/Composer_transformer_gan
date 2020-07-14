import os
import glob
import numpy as np

from keras.layers import Dense, Dropout, Input, Conv2D, BatchNormalization, add, \
    Conv2DTranspose, GaussianNoise, AveragePooling2D, Flatten
from keras.models import Model, Sequential, load_model
from keras.layers.advanced_activations import ReLU, LeakyReLU, ELU
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import pickle as pkl

from convertor import Conversion


# GAN ==============================
class Generator:

    def __init__(self, input_dim=500, filter_list=(8, 6, 4), strides_list=((2, 11), (2, 4), (1, 2)),
                 kernel_list=(5, 4, 3), resnet_up_add_noise=False, upsample_add_noise=True):

        x_in = Input(shape=(input_dim, 1, 1))
        x = x_in
        # upsampling
        for fltr, strd, knl in zip(filter_list, strides_list, kernel_list):
            # upsampling
            x = Generator.upsampling_block(x, fltr, knl, strd, 'same', upsample_add_noise)
            # residual block
            x = Generator.residual_block(x, fltr, knl, resnet_up_add_noise)
        # output
        out = Conv2D(filters=1, kernel_size=kernel_list[-1], activation='sigmoid', padding='same')(x)
        # model
        self.model = Model(inputs=[x_in], outputs=[out])

    @staticmethod
    def residual_block(x, fltr, kernel_size, add_noise=True):
        x0 = Conv2D(filters=fltr, kernel_size=kernel_size, activation=None, padding='same')(x)
        x0 = BatchNormalization(momentum=0.6)(x0)
        if add_noise:
            x0 = GaussianNoise(0.1)(x0)
        x1 = ELU()(x0)
        x1 = Conv2D(filters=fltr, kernel_size=kernel_size, activation=None, padding='same')(x1)
        x1 = BatchNormalization(momentum=0.6)(x1)
        x1 = ELU()(x1)
        x1 = Conv2D(filters=fltr, kernel_size=kernel_size, activation=None, padding='same')(x1)
        x1 = BatchNormalization(momentum=0.6)(x1)
        x1 = add([x1, x0])
        x1 = ELU()(x1)
        return x1

    @staticmethod
    def upsampling_block(x, fltr, kernel_size, strides=(2, 2), padding='same', add_noise=True):
        if add_noise:
            x = GaussianNoise(0.1)(x)
        x = Conv2DTranspose(fltr, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(momentum=0.6)(x)
        return x


class Discriminator:

    def __init__(self, input_dim=2000, filter_list=(4, 6, 8), kernel_list=(3, 4, 5), pool=True, dropout=0.2):
        x_in = Input(shape=(input_dim, 88, 1))
        x = x_in
        for fltr, knl in zip(filter_list, kernel_list):
            x = Discriminator.discriminator_block(x, fltr, knl, pool=pool, dropout=dropout)
        x = Flatten()(x)
        out = Dense(1, kernel_initializer='he_normal', activation='sigmoid')(x)

        # model
        self.model = Model(inputs=[x_in], outputs=[out])

    @staticmethod
    def discriminator_block(x, fltr, kernel_size, pool=True, dropout=0.2):
        x = Conv2D(filters=fltr, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                   bias_initializer='zeros')(x)
        x = LeakyReLU(0.01)(x)
        if pool:
            x = AveragePooling2D()(x)
        x = Conv2D(filters=fltr, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                   bias_initializer='zeros')(x)
        x = LeakyReLU(0.01)(x)
        if dropout is not None:
            x = Dropout(0.2)(x)
        return x


class GAN:

    def __init__(self, g_input_dim=500, g_filter_list=(8, 6, 4), g_strides_list=((2, 11), (2, 4), (1, 2)),
                 g_kernel_list=(5, 4, 3), g_resnet_up_add_noise=False, g_upsample_add_noise=True,
                 g_model_path_name=None, g_loaded_trainable=False,
                 d_input_dim=2000, d_filter_list=(4, 6, 8), d_kernel_list=(3, 4, 5), d_pool=True, d_dropout=0.2,
                 d_model_path_name=None, d_loaded_trainable=False,
                 d_lr=0.01, d_clipnorm=1.0, f_lr=0.01, f_clipnorm=1.0,
                 pixmax=127):
        self.g_input_dim = g_input_dim
        self.pixmax = pixmax

        # generator model
        self.gmodel = Generator(
            input_dim=g_input_dim, filter_list=g_filter_list, strides_list=g_strides_list, kernel_list=g_kernel_list,
            resnet_up_add_noise=g_resnet_up_add_noise, upsample_add_noise=g_upsample_add_noise).model
        if g_model_path_name is not None:
            self.gmodel = self.load_existing_model(self.gmodel, g_model_path_name, g_loaded_trainable)

        # discriminator model
        self.dmodel = Discriminator(
            input_dim=d_input_dim, filter_list=d_filter_list, kernel_list=d_kernel_list, pool=d_pool,
            dropout=d_dropout).model
        if d_model_path_name is not None:
            self.dmodel = self.load_existing_model(self.dmodel, d_model_path_name, d_loaded_trainable)
        self.dmodel.compile(
            loss='binary_crossentropy', optimizer=Adam(lr=d_lr, clipnorm=d_clipnorm), metrics=['accuracy'])

        # full model
        self.dmodel.trainable = False  # freeze discriminator while training the full model
        self.full_model = Sequential()
        self.full_model.add(self.gmodel)
        self.full_model.add(self.dmodel)
        self.full_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=f_lr, clipnorm=f_clipnorm))

    @staticmethod
    def cut_array(ary, length=2000, n_pieces=5):
        ary_len = ary.shape[0]
        start_inds = np.random.uniform(low=0, high=ary_len - length, size=n_pieces).astype(int)
        return [ary[id_: (id_ + length)].tolist() for id_ in start_inds]

    def load_data(self, piece_length=2000, data_path='/Users/Wei/Desktop/piano_classic/Chopin_array',
                  name_contain='*noct*'):
        all_array_names = glob.glob(os.path.join(data_path, name_contain))
        self.true_data = []
        for nm in all_array_names:
            tmp = GAN.cut_array(pkl.load(open(nm, 'rb')), length=piece_length, n_pieces=5)
            self.true_data += tmp
        self.true_data = np.expand_dims(np.array(self.true_data) / self.pixmax, axis=3)

    def true_samples(self, n_samples=20):
        inds = np.random.randint(0, self.true_data.shape[0], n_samples)
        return self.true_data[inds]

    def fake_samples(self, n_samples=20):
        in_samples = np.random.normal(0.0, 1.0, size=[n_samples, self.g_input_dim, 1, 1])
        return self.gmodel.predict(in_samples)

    def load_existing_model(self, model, load_model_path_name, loaded_trainable):
        updated_model = load_model(load_model_path_name)
        for i in range(len(updated_model.layers)):
            model.layers[i] = updated_model.layers[i]
            model.layers.trainable = loaded_trainable
        return model

    def train(self, n_epoch=100, n_samples=20, save_step=10, n_save=2, verbose=True, save_as_mid=False, save_pic=True,
              save_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/result'):
        true_y, fake_y = np.ones((n_samples, 1), dtype=np.float32), np.zeros((n_samples, 1), dtype=np.float32)
        self.d_perform, self.f_perform = [], []
        for i in range(n_epoch):
            # generate fake samples
            fake_x = self.fake_samples(n_samples)  # (n_samples, d_input_dim, 88, 1)
            true_x = self.true_samples(n_samples)  # (n_samples, d_input_dim, 88, 1)
            combine_x = np.array(fake_x.tolist()+true_x.tolist())
            combine_y = np.array(fake_y.tolist()+true_y.tolist())

            # train discriminator on combined samples
            d_loss, d_metr = self.dmodel.train_on_batch(combine_x, combine_y)
            self.d_perform.append([d_loss, d_metr])

            # generate latent vector input
            latent_x = np.random.normal(0.0, 1.0, size=[n_samples*2, self.g_input_dim, 1, 1])
            f_loss = self.full_model.train_on_batch(latent_x, np.zeros((n_samples*2, 1), dtype=np.float32))

            # print result
            if verbose:
                print('epoch {}: d_loss={}, d_metr={}, f_loss={}'.format(i+1, d_loss, d_metr, f_loss))

            # save predictions
            if (i > 0) & (i % save_step == 0):
                print('save midi')
                fake_samples_save = (np.squeeze(fake_x[:n_save], axis=3) * self.pixmax).astype(int)  # (n_save, 2000, 88)
                for j, f_s in enumerate(fake_samples_save):
                    file_name = '{}_{}'.format(i, j)
                    if save_as_mid:
                        mid_new = Conversion.arry2mid(f_s, 500000)
                        mid_new.save(os.path.join(save_path, file_name+'.mid'))
                    else:
                        pkl.dump(f_s, open(os.path.join(save_path, file_name+'.pkl'), 'wb'))
                    if save_pic:
                        plt.figure(figsize=(20, 5))
                        plt.plot(range(f_s.shape[0]), np.multiply(np.where(f_s > 0, 1, 0), range(1, 89)), marker='.',
                                 markersize=1, linestyle='')
                        plt.savefig(os.path.join(save_path, file_name + '.png'))
                        plt.close()

