import os
import glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pickle as pkl
import tensorflow as tf
import util
import transformer
import discriminator
import generator

notes_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_embedder'
notes_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_generator'
time_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_generator'
notes_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_discriminator'
time_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_discriminator'
discriminator_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/discriminator'

tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_indexcer/notes_dict_final.pkl'
tk = pkl.load(open(tk_path, 'rb'))


class GAN(tf.keras.Model):

    def __init__(self, strt_token_id=15001, out_notes_pool_size=15002, embed_dim=256, n_heads=4, max_pos=800,
                 time_features=3,
                 fc_activation="relu",
                 g_encoder_layers=2, g_decoder_layers=2, g_fc_layers=3, g_norm_epsilon=1e-6,
                 g_embedding_dropout_rate=0.2, g_transformer_dropout_rate=0.2,

                 d_kernel_size=3, d_encoder_layers=2, d_decoder_layers=2, d_fc_layers=3, d_norm_epsilon=1e-6,
                 d_transformer_dropout_rate=0.2,

                 notes_latent_nlayers=4, notes_latent_dim_base=4, time_latent_nlayers=4,

                 mode_='both', custm_lr=True,

                 lr_tm=0.01, warmup_steps=4000,
                 optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)):
        super(GAN, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'

        # ---------------------------------- settings ----------------------------------
        self.mode_ = mode_  # only choose from ['notes', 'time', 'both']
        self.embed_dim = embed_dim
        if custm_lr:
            learning_rate = util.CustomSchedule(embed_dim, warmup_steps)
        else:
            learning_rate = lr_tm
        self.optimizer = optmzr(learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.strt_token_id = strt_token_id  # strt_token_id = tk.word_index['<start>']

        # ---------------------------------- layers ----------------------------------
        # latent vector generator
        self.notes_latent = generator.NotesLatent(nlayers=notes_latent_nlayers, dim_base=notes_latent_dim_base) \
            if mode_ in ['notes', 'both'] else None
        self.time_latent = generator.TimeLatent(nlayers=time_latent_nlayers) if mode_ in ['time', 'both'] else None
        # music generator
        self.gen = generator.Generator(
            out_notes_pool_size=out_notes_pool_size, embed_dim=embed_dim, n_heads=n_heads, max_pos=max_pos,
            time_features=time_features, fc_activation=fc_activation, encoder_layers=g_encoder_layers,
            decoder_layers=g_decoder_layers, fc_layers=g_fc_layers, norm_epsilon=g_norm_epsilon,
            embedding_dropout_rate=g_embedding_dropout_rate, transformer_dropout_rate=g_transformer_dropout_rate,
            mode_=mode_)
        # discriminator
        self.disc = discriminator.Discriminator(
            embed_dim=embed_dim, n_heads=n_heads, kernel_size=d_kernel_size, fc_activation=fc_activation,
            encoder_layers=d_encoder_layers, decoder_layers=d_decoder_layers, fc_layers=d_fc_layers,
            norm_epsilon=d_norm_epsilon, transformer_dropout_rate=d_transformer_dropout_rate, mode_='both')

    def call(self, inputs, training=None, mask=None):
        # generate music from latent
        #tf.constant([[tk.word_index['<start>']]] * batch_size, dtype=tf.float32)
        pass

    def train_step(self, nt_ltnt, tm_ltnt, out_seq_len, freeze_disc=True):
        # nt_ltnt: notes random vector (batch, in_seq_len, 16)
        # tm_ltnt: time random vector (batch, in_seq_len, 1)

        with tf.GradientTape() as tape:

            # todo: freeze discriminator ?


            # ---------------------- create latent vectors from random inputs ----------------------
            # (batch, in_seq_len, embed_dim)
            nt_ltnt_ = self.notes_latent(nt_ltnt) if self.mode_ in ['notes', 'both'] else None
            # (batch, in_seq_len, 3)
            tm_ltnt_ = self.time_latent(tm_ltnt) if self.mode_ in ['time', 'both'] else None

            # ---------------------- generate music from latent vectors ----------------------
            if self.mode_ == 'notes':
                # get nts: (batch, out_seq_len)
                nts = self.gen(x_en_nt=nt_ltnt_, x_en_tm=tm_ltnt_, tk=tk, out_seq_len=out_seq_len, return_str=False,
                               vel_norm=None, tmps_norm=None, dur_norm=None, return_denorm=False)
                nts = tf.convert_to_tensor(nts, dtype=tf.float32)  # convert from numpy to tensor
            elif self.mode_ == 'time':
                # get tms: (batch, out_seq_len, 3)
                tms = self.gen(x_en_nt=nt_ltnt_, x_en_tm=tm_ltnt_, tk=tk, out_seq_len=out_seq_len, return_str=False,
                               vel_norm=None, tmps_norm=None, dur_norm=None, return_denorm=False)
                tms = tf.convert_to_tensor(tms, dtype=tf.float32)  # convert from numpy to tensor
            else:  # self.mode_ == 'both'
                # get nts: (batch, out_seq_len)
                # get tms: (batch, out_seq_len, 3)
                nts, tms = self.gen(x_en_nt=nt_ltnt_, x_en_tm=tm_ltnt_, tk=tk, out_seq_len=out_seq_len, return_str=False,
                                    vel_norm=None, tmps_norm=None, dur_norm=None, return_denorm=False)
                nts = tf.convert_to_tensor(nts, dtype=tf.float32)  # convert from numpy to tensor
                tms = tf.convert_to_tensor(tms, dtype=tf.float32)  # convert from numpy to tensor

            # ---------------------- generated notes embedding ----------------------
            # get nts: (batch, out_seq_len, embed_dim)
            if self.mode_ in ['notes', 'both']:
                nts = self.gen.notes_emb(nts)

            # ---------------------- prepare samples ----------------------
            #fake_samples =






            # ---------------------- discriminator ----------------------

            #
            # strt_token_id
            #
            # self.disc(nts, tms, de_in)













    def load_model(self):
        pass



    def train(self,
              freeze_ntgen=False, freeze_tmgen=False, freeze_ntemb=False, freeze_ntlatent=False, freeze_tmlatent=False):

        # Todo: call self.load_model

        # # ---------------------- call back setting --------------------------
        # if self.mode_ in ['notes', 'both']:
        #     cp_notes_emb = tf.train.Checkpoint(model=self.notes_emb, optimizer=self.optimizer)
        #     cp_manager_notes_emb = tf.train.CheckpointManager(cp_notes_emb, notes_emb_path, max_to_keep=max_to_keep)
        #     if cp_manager_notes_emb.latest_checkpoint:
        #         cp_notes_emb.restore(cp_manager_notes_emb.latest_checkpoint)
        #         print('Restored the latest notes_emb')
        #
        #     cp_notes_gen = tf.train.Checkpoint(model=self.notes_gen, optimizer=self.optimizer)
        #     cp_manager_notes_gen = tf.train.CheckpointManager(cp_notes_gen, notes_gen_path, max_to_keep=max_to_keep)
        #     if cp_manager_notes_gen.latest_checkpoint:
        #         cp_notes_gen.restore(cp_manager_notes_gen.latest_checkpoint)
        #         print('Restored the latest notes_gen')
        #
        # if self.mode_ in ['time', 'both']:
        #     cp_time_gen = tf.train.Checkpoint(model=self.time_gen, optimizer=self.optimizer)
        #     cp_manager_time_gen = tf.train.CheckpointManager(cp_time_gen, time_gen_path, max_to_keep=max_to_keep)
        #     if cp_manager_time_gen.latest_checkpoint:
        #         cp_time_gen.restore(cp_manager_time_gen.latest_checkpoint)
        #         print('Restored the latest time_gen')



        # also set for discriminator, notes_discriminator, time_discriminator
        pass


"""
from tensorflow.keras.layers import Input

out_notes_pool_size=15002
embed_dim=256
n_heads=4
max_pos=800
time_features=3
fc_activation="relu"
g_encoder_layers=2
g_decoder_layers=2
g_fc_layers=3
g_norm_epsilon=1e-6
g_embedding_dropout_rate=0.2
g_transformer_dropout_rate=0.2
d_kernel_size=3
d_encoder_layers=2
d_decoder_layers=2
d_fc_layers=3
d_norm_epsilon=1e-6
d_transformer_dropout_rate=0.2
notes_latent_nlayers=4
notes_latent_dim_base=4
time_latent_nlayers=4
mode_='both'
custm_lr=True
freeze_ntgen=False
freeze_tmgen=False
freeze_ntemb=False
freeze_ntlatent=False
freeze_tmlatent=False
lr_tm=0.01
warmup_steps=4000
optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
strt_token_id=15001
"""




# # https://www.tensorflow.org/tutorials/text/transformer
#
#
# import os
# import glob
# import numpy as np
# import itertools
# import matplotlib.pyplot as plt
# import pickle as pkl
# from scipy import sparse
#
# import tensorflow as tf
#
# import util
# import transformer
# import convertor
# import discriminator
# import generator
#
#
# class LatentVector(tf.keras.Model):
#
#     def __init__(self, embed_dim=256, nlayers=4, dim_base=4):
#         # generate latent vector of dimension; (batch, in_seq_len, embed_dim)
#         super(LatentVector, self).__init__()
#         self.embed_dim = embed_dim
#         self.nlayers = nlayers
#         self.fcs = tf.keras.models.Sequential(
#             [tf.keras.layers.Dense(dim_base**i, use_bias=True) for i in range(1, nlayers+1)])
#
#     def call(self, x_in):
#         # x_in: (batch, in_seq_len, 1)
#         return self.fcs(x_in)
#
#
# class GAN:
#
#     def __init__(self, g_in_seq_len=16, latent_nlayers=4, latent_dim_base=4,
#                  g_out_seq_len=64,
#                  g_de_max_pos=10000, embed_dim=256, n_heads=4,
#                  g_out_notes_pool_size=8000, g_fc_activation="relu", g_encoder_layers=2, g_decoder_layers=2,
#                  g_fc_layers=3,
#                  g_norm_epsilon=1e-6, g_transformer_dropout_rate=0.2, g_embedding_dropout_rate=0.2, beta=1e10,
#                  loss='mean_squared_error', metrics=['mae'], opt=tf.keras.optimizers.Adam(lr=0.1, clipnorm=1.0),
#                  ):
#
#         # g_de_init: tf.expand_dims(tf.constant([[tk.word_index['<start>'], 0]] * batch_size, dtype=tf.float32), 1)
#         g_de_init = tf.keras.layers.Input((1, 2))
#         latent_melody = tf.keras.layers.Input((g_in_seq_len, 1))
#
#         # ========================== Generator ==========================
#
#         # generate latent vector -------------------------
#         # x_en_melody: (batch, g_in_seq_len, embed_dim)
#         # x_en_duration: (batch, g_in_seq_len, 1)
#         x_en_melody = LatentVector(embed_dim=embed_dim, nlayers=latent_nlayers, dim_base=latent_dim_base)(
#             latent_melody)
#         x_en_duration = LatentVector(embed_dim=1, nlayers=latent_nlayers, dim_base=1)(
#             latent_melody)
#
#         # generate latent vector -------------------------
#         gen = generator.Generator(
#             de_max_pos=g_de_max_pos, embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=g_out_notes_pool_size,
#             fc_activation=g_fc_activation, encoder_layers=g_encoder_layers, decoder_layers=g_decoder_layers,
#             fc_layers=g_fc_layers, norm_epsilon=g_norm_epsilon, transformer_dropout_rate=g_transformer_dropout_rate,
#             embedding_dropout_rate=g_embedding_dropout_rate)
#         g_out = GAN.generate_music(
#             x_en_melody=x_en_melody, x_en_duration=x_en_duration, g_de_init=g_de_init,
#             g_out_seq_len=g_out_seq_len, gen=gen, beta=beta)
#
#
#
#
#
#
#
#         #self.gmodel = tf.keras.models.Model(inputs=[latent_in, g_de_init], outputs=g_out)
#
#         # Todo: gmodel.summary() shows Total params: 0, Trainable params: 0, Non-trainable params: 0
#
#         # tmp = Model(inputs=latent_in, outputs=g_in)
#         # tmp.summary()
#
#
#         # gmodel.compile(loss=loss, optimizer=opt, metrics=metrics)
#
#         # ========================== Discriminator ==========================
#
#         #x_en_melody, x_de_in, mask_padding, mask_lookahead
#
#         # ---------------- generator encoder embedder --------------------
#         #self.g_encoder_embedder =
#
#         # self.g_input_dim = g_input_dim
#         # self.latent_mean = latent_mean
#         # self.latent_std = latent_std
#
#         # generator model
#         # g_x_in = Input(shape=(g_input_dim, 1))
#         # g_x_out = Blocks.generator_block(
#         #     g_x_in, fc_dim=g_fc_dim, fltr_dims=g_fltr_dims, knl_dims=g_knl_dims, up_dims=g_up_dims,
#         #     bn_momentum=g_bn_momentum, noise_std=g_noise_std, final_fltr=g_final_fltr, final_knl=g_final_knl)
#         # self.gmodel = Model(inputs=[g_x_in], outputs=[g_x_out])
#         # if g_model_path_name is not None:
#         #     self.gmodel = self.load_existing_model(self.gmodel, g_model_path_name, g_loaded_trainable, g_loss)
#         # if print_model:
#         #     print(self.gmodel.summary())
#         #
#         # # discriminator model
#         # d_x_in = Input(shape=(self.gmodel.output.shape[1], 1))
#         # d_x_out = Blocks.discriminator_block(
#         #     d_x_in, fltr_dims=d_fltr_dims, knl_dims=d_knl_dims, down_dims=d_down_dims, bn_momentum=d_bn_momentum)
#         # self.dmodel = Model(inputs=[d_x_in], outputs=[d_x_out])
#         # if d_model_path_name is not None:
#         #     self.dmodel = self.load_existing_model(self.dmodel, d_model_path_name, d_loaded_trainable)
#         # if print_model:
#         #     print(self.dmodel.summary())
#         # self.dmodel.compile(loss=d_loss, optimizer=Adam(lr=d_lr, clipnorm=d_clipnorm), metrics=d_metrics)
#         #
#         # # full model
#         # self.dmodel.trainable = False  # freeze discriminator while training the full model
#         # gan_out = self.dmodel(g_x_out)
#         # self.full_model = Model(inputs=[g_x_in], outputs=[gan_out])
#         # if print_model:
#         #     print(self.full_model.summary())
#         # self.full_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=f_lr, clipnorm=f_clipnorm))
#
#     @staticmethod
#     def generate_music(x_en_melody, x_en_duration, g_de_init, g_out_seq_len, gen, beta=1e10):
#         # gen is the generator model
#         # x_en_melody: (batch, g_in_seq_len, embed_dim)
#         # x_en_duration: (batch, g_in_seq_len, 1)
#         # g_de_init: (batch, 1, 2):
#         #       tf.expand_dims(tf.constant([[tk.word_index['<start>'], 0]] * batch_size, dtype=tf.float32), 1)
#         de_in = g_de_init
#         for i in range(g_out_seq_len):
#             # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
#             # x_out_duration: (batch, out_seq_len, 1)
#             mask_lookahead = util.lookahead_mask(i+1)  # (len(x_de_in), len(x_de_in))
#             x_out_melody, all_weights_melody, x_out_duration, all_weights_duration = gen(
#                 x_en_melody=x_en_melody, x_en_duration=x_en_duration, x_de_in=de_in, mask_padding=None,
#                 mask_lookahead=mask_lookahead)
#             melody_ = util.softargmax(x_out_melody, beta=beta)  # (batch, None)
#             melody_ = tf.expand_dims(melody_, axis=-1)  # (None, None, 1)
#             music_ = tf.concat([melody_, x_out_duration], axis=-1)  # (None, None, 2): column: notes_id, duration
#             de_in = tf.concat((de_in, music_), axis=1)
#         return de_in
#
#

#
#
#     @staticmethod
#     def true_samples(all_true_x, n_samples=20):
#         inds = np.random.randint(0, all_true_x.shape[0], n_samples)
#         true_x = all_true_x[inds]
#         return true_x
#
#     def fake_samples(self, n_samples=20):
#         x_in = DataPreparation.latant_vector(n_samples, self.g_input_dim, mean_=self.latent_mean, std_=self.latent_std)
#         return self.gmodel.predict(x_in)
#
#     def load_existing_model(self, model, load_model_path_name, loaded_trainable, loss_func):
#         updated_model = load_model(load_model_path_name) if loss_func is None \
#             else load_model(load_model_path_name, custom_objects={'loss_func': loss_func})
#         for i in range(len(updated_model.layers)):
#             model.layers[i] = updated_model.layers[i]
#             model.layers[i].trainable = loaded_trainable
#         return model
#
#     def train(self, all_true_x, n_epoch=100, n_samples=20, save_step=10, n_save=1, verbose=True, save_pic=True,
#               file_save_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/result',
#               model_save_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/models'):
#         true_y, fake_y = np.ones((n_samples, 1), dtype=np.float32), np.zeros((n_samples, 1), dtype=np.float32)
#         self.d_perform, self.f_perform = [], []
#         for i in range(n_epoch):
#             true_x = GAN.true_samples(all_true_x, n_samples=n_samples)
#             fake_x = self.fake_samples(n_samples=n_samples)
#             combine_x = np.array(fake_x.tolist()+true_x.tolist())
#             combine_y = np.array(fake_y.tolist()+true_y.tolist())
#
#             # train discriminator on combined samples
#             d_loss, d_metr = self.dmodel.train_on_batch(combine_x, combine_y)
#             self.d_perform.append([d_loss, d_metr])
#
#             # generate latent vector input
#             latent_x = DataPreparation.latant_vector(
#                 n_samples*2, self.g_input_dim, mean_=self.latent_mean, std_=self.latent_std)
#             f_loss = self.full_model.train_on_batch(latent_x, np.ones((n_samples*2, 1), dtype=np.float32))  # inverse label!!!!
#             # print result
#             if verbose:
#                 print('epoch {}: d_loss={}, d_metr={}, f_loss={}'.format(i+1, d_loss, d_metr, f_loss))
#
#             # save predictions
#             if (i > 0) & (i % save_step == 0):
#                 print('save result')
#                 self.gmodel.save(os.path.join(model_save_path, 'gmodel.h5'))
#                 self.dmodel.save(os.path.join(model_save_path, 'dmodel.h5'))
#                 fake_samples_save = fake_x[:n_save]
#                 for j, f_s in enumerate(fake_samples_save):
#                     file_name = '{}_{}'.format(i, j)
#                     f_s = DataPreparation.recover_array(f_s)
#                     pkl.dump(sparse.csr_matrix(f_s), open(os.path.join(file_save_path, file_name+'.pkl'), 'wb'))
#                     if save_pic:
#                         plt.figure(figsize=(20, 5))
#                         plt.plot(range(f_s.shape[0]), np.multiply(f_s, range(1, 89)), marker='.',
#                                  markersize=1, linestyle='')
#                         plt.savefig(os.path.join(file_save_path, file_name + '.png'))
#                         plt.close()
#
#
# """
# filepath = '/Users/Wei/Desktop/piano_classic/Chopin_array/nocturne_c_sharp-_(c)unknown.pkl'
#
# g_in_seq_len=16
# latent_nlayers=4
# latent_dim_base=4
# g_out_seq_len=64
# g_de_max_pos=10000
# embed_dim=256
# n_heads=4
# g_out_notes_pool_size=8000
# g_fc_activation="relu"
# g_encoder_layers=2
# g_decoder_layers=2
# g_fc_layers=3
# g_norm_epsilon=1e-6
# g_transformer_dropout_rate=0.2
# g_embedding_dropout_rate=0.2
# beta=1e10
# """
