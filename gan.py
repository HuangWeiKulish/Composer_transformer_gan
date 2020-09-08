import os
import glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pickle as pkl
import tensorflow as tf
import util
import discriminator
import generator
import time

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
                 out_seq_len=64,

                 mode_='both', custm_lr=True,

                 lr_tm=0.01, warmup_steps=4000,
                 optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)):
        super(GAN, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'

        # ---------------------------------- settings ----------------------------------
        self.batch_size = None
        self.true_data = None
        self.in_seq_len = 16
        self.out_seq_len = out_seq_len
        self.strt_token_id = strt_token_id  # strt_token_id = tk.word_index['<start>']
        self.mode_ = mode_  # only choose from ['notes', 'time', 'both']
        self.embed_dim = embed_dim
        if custm_lr:
            learning_rate = util.CustomSchedule(embed_dim, warmup_steps)
        else:
            learning_rate = lr_tm
        self.optimizer = optmzr(learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

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
            norm_epsilon=d_norm_epsilon, transformer_dropout_rate=d_transformer_dropout_rate, mode_=mode_)

    def load_true_samples(self, tk, step=30, batch_size=10, vel_norm=64.0, tmps_norm=0.12,
                          dur_norm=1.3, pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['']):
        self.batch_size = batch_size
        self.true_data = util.load_true_data_gan(
            tk, self.out_seq_len, step=step, batch_size=batch_size, vel_norm=vel_norm,
            tmps_norm=tmps_norm, dur_norm=dur_norm, pths=pths, name_substr_list=name_substr_list)

    def call(self, inputs, training=None, mask=None):
        # generate music from latent

        #tf.constant([[tk.word_index['<start>']]] * batch_size, dtype=tf.float32)
        pass

    def model_trainable(self, model, trainable=True):
        pass

    def prepare_fake_samples(self, nt_ltnt, tm_ltnt):
        # ---------------------- create latent vectors from random inputs ----------------------
        # (batch, in_seq_len, embed_dim)
        nt_ltnt_ = self.notes_latent(nt_ltnt) if self.mode_ in ['notes', 'both'] else None
        # (batch, in_seq_len, 3)
        tm_ltnt_ = self.time_latent(tm_ltnt) if self.mode_ in ['time', 'both'] else None

        # ---------------------- generate music from latent vectors ----------------------
        if self.mode_ == 'notes':
            # get nts: (batch, out_seq_len)
            nts = self.gen(x_en_nt=nt_ltnt_, x_en_tm=tm_ltnt_, tk=tk, out_seq_len=self.out_seq_len,
                           return_str=False, vel_norm=None, tmps_norm=None, dur_norm=None, return_denorm=False)
            nts = tf.convert_to_tensor(nts, dtype=tf.float32)  # convert from numpy to tensor
            nts = self.gen.notes_emb(nts)
            return nts

        if self.mode_ == 'time':
            # get tms: (batch, out_seq_len, 3)
            tms = self.gen(x_en_nt=nt_ltnt_, x_en_tm=tm_ltnt_, tk=tk, out_seq_len=self.out_seq_len,
                           return_str=False, vel_norm=None, tmps_norm=None, dur_norm=None, return_denorm=False)
            tms = tf.convert_to_tensor(tms, dtype=tf.float32)  # convert from numpy to tensor
            return tms

        # self.mode_ == 'both'
        # get nts: (batch, out_seq_len) index representation
        # get tms: (batch, out_seq_len, 3) NOT denormalized
        nts, tms = self.gen(x_en_nt=nt_ltnt_, x_en_tm=tm_ltnt_, tk=tk, out_seq_len=self.out_seq_len,
                            return_str=False, vel_norm=None, tmps_norm=None, dur_norm=None, return_denorm=False)
        nts = tf.convert_to_tensor(nts, dtype=tf.float32)  # convert from numpy to tensor
        nts = self.gen.notes_emb(nts)
        tms = tf.convert_to_tensor(tms, dtype=tf.float32)  # convert from numpy to tensor
        return nts, tms

    def train_discriminator(self, nt_ltnt, tm_ltnt, nts_tr, tms_tr):
        # nt_ltnt: (batch, in_seq_len, 16)
        # tm_ltnt: (batch, in_seq_len, 1)
        # nts_tr: (batch, out_seq_len)
        # tms_tr: (batch, out_seq_len, 3)

        # Todo: unfreeze discriminator

        # prepare fake samples ------------------------------------------------------------------
        # nts_fk: (batch, out_seq_len, embed_dim)
        # tms_fk: (batch, out_seq_len, 3)
        if self.mode_ == 'notes':
            tms_fk = None
            nts_fk = self.prepare_fake_samples(nt_ltnt, tm_ltnt)
        elif self.mode_ == 'time':
            tms_fk = self.prepare_fake_samples(nt_ltnt, tm_ltnt)
            nts_fk = None
        else:  # self.mode_ == 'both'
            nts_fk, tms_fk = self.prepare_fake_samples(nt_ltnt, tm_ltnt)

        # prepare true samples ------------------------------------------------------------------
        # nts_tr: (batch, out_seq_len, embed_dim)
        if self.mode_ in ['notes', 'both']:
            nts_tr = self.gen.notes_emb(nts_tr)

        # prepare decode initial input ------------------------------------------------------------------
        # de_in: (batch, 1, embed_dim)
        de_in = self.gen.notes_emb(tf.constant([[0]] * self.batch_size * 2, dtype=tf.float32)) \
            if self.mode_ == 'time' \
            else self.gen.notes_emb(tf.constant([[self.strt_token_id]] * self.batch_size, dtype=tf.float32))

        # combine samples ------------------------------------------------------------------
        # nts_comb: (bacth * 2, out_seq_len, embed_dim)
        # tms_comb: (bacth * 2, out_seq_len, 3)
        # lbl_comb: (bacth * 2, 1)
        nts_comb = tf.concat([nts_fk, nts_tr], axis=0) if self.mode_ in ['notes', 'both'] else None
        tms_comb = tf.concat([tms_fk, tms_tr], axis=0) if self.mode_ in ['time', 'both'] else None
        lbl_comb = tf.constant([[0]] * self.batch_size + [[1]] * self.batch_size, dtype=tf.float32)

        # predict true or fake sample ------------------------------------------------------------------
        pred_comb = self.disc(nts_comb, tms_comb, de_in)
        return pred_comb, lbl_comb

    def train_generator(self, nt_ltnt, tm_ltnt):
        # nt_ltnt: (batch, in_seq_len, 16)
        # tm_ltnt: (batch, in_seq_len, 1)

        # Todo: freeze discriminator

        # prepare fake samples ------------------------------------------------------------------
        # nts_fk: (batch, out_seq_len, embed_dim)
        # tms_fk: (batch, out_seq_len, 3)
        if self.mode_ == 'notes':
            nts_fk = self.prepare_fake_samples(nt_ltnt, tm_ltnt)
            tms_fk = None
        elif self.mode_ == 'time':
            nts_fk = None
            tms_fk = self.prepare_fake_samples(nt_ltnt, tm_ltnt)
        else:  # self.mode_ == 'both'
            nts_fk, tms_fk = self.prepare_fake_samples(nt_ltnt, tm_ltnt)

        # prepare decode initial input ------------------------------------------------------------------
        # de_in: (batch, 1, embed_dim)
        de_in = self.gen.notes_emb(tf.constant([[0]] * self.batch_size, dtype=tf.float32)) \
            if self.mode_ == 'time' \
            else self.gen.notes_emb(tf.constant([[self.strt_token_id]] * self.batch_size, dtype=tf.float32))

        # combine samples ------------------------------------------------------------------
        # lbl_fk: (bacth, 1)
        lbl_fk = tf.constant([[1]] * self.batch_size, dtype=tf.float32)

        # predict true or fake sample ------------------------------------------------------------------
        pred_fk = self.disc(nts_fk, tms_fk, de_in)
        return pred_fk, lbl_fk






    def train_step(self, nt_ltnt, tm_ltnt, nts_tr, tms_tr, freeze_disc=True):
        # nt_ltnt: notes random vector (batch, out_seq_len, 16)
        # tm_ltnt: time random vector (batch, out_seq_len, 1)

        with tf.GradientTape() as tape:




            # todo: freeze discriminator ?
            # Todo: 1. train on discriminator with combined true and fake samples;
            # Todo: 2. generate another batch * 2 of fake samples and train on combined models, with all label True

            pass

        #     loss_notes = util.loss_func_notes(x_tar_out[:, :, 0], x_out_nt_pr)
        #     variables_notes = self.notes_emb.trainable_variables + self.notes_gen.trainable_variables
        #
        # if self.mode_ in ['time', 'both']:
        #     #  x_in_tm: (batch, in_seq_len, 3)
        #     #  x_tar_in_tm: (batch, out_seq_len, 3)
        #     #  x_tar_out_tm: (batch, out_seq_len, 3)
        #     x_in_tm, x_tar_in_tm, x_tar_out_tm = x_in[:, :, 1:], x_tar_in[:, :, 1:], x_tar_out[:, :, 1:]
        #     # x_out_nt_pr: (batch, out_seq_len, 3)
        #     x_out_tm_pr, _ = self.time_gen(
        #         x_en=x_in_tm, x_de=x_tar_in_tm, mask_padding=mask_padding, mask_lookahead=mask_lookahead)
        #     # the ending value is 0, remove from loss calcultion
        #     loss_time = util.loss_func_time(x_tar_out_tm[:, :-1, :], x_out_tm_pr[:, :-1, :])
        #     variables_time = self.time_gen.trainable_variables
        #
        # if self.mode_ == 'notes':
        #     gradients = tape.gradient(loss_notes, variables_notes)
        #     self.optimizer.apply_gradients(zip(gradients, variables_notes))
        #     self.train_loss(loss_notes)
        #     return loss_notes
        # elif self.mode_ == 'time':
        #     gradients = tape.gradient(loss_time, variables_time)
        #     self.optimizer.apply_gradients(zip(gradients, variables_time))
        #     self.train_loss(loss_time)
        #     return loss_time
        # else:  # self.mode_ == 'both'
        #     variables_combine = variables_notes + variables_time
        #     loss_combine = (loss_notes * nt_tm_loss_weight[0] +
        #                     loss_time * nt_tm_loss_weight[1]) / sum(nt_tm_loss_weight)
        #     gradients = tape.gradient(loss_combine, variables_combine)
        #     self.optimizer.apply_gradients(zip(gradients, variables_combine))
        #     self.train_loss(loss_combine)
        #     return loss_notes, loss_time, loss_combine

    def load_model(self):
        pass



    def train(self, epochs=10, nt_tm_loss_weight=(1, 1), save_model_step=1,
              notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path, time_gen_path=time_gen_path,
              max_to_keep=5, print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=5,
              lr_tm=0.01, warmup_steps=4000, custm_lr=True,
              optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
              freeze_ntgen=False, freeze_tmgen=False, freeze_ntemb=False, freeze_ntlatent=False, freeze_tmlatent=False):

        # Todo: call self.load_model

        # # ---------------------- call back setting --------------------------
        # also set for discriminator, notes_discriminator, time_discriminator




        # ---------------------- training --------------------------
        for epoch in range(epochs):
            self.train_loss.reset_states()
            start = time.time()
            for i, true_samples in enumerate(self.true_data):

                if true_samples.shape[0] < self.batch_size:
                    #  the last batch generated may be smaller, so go to next round
                    # the unselected samples may be selected next time because data will be shuffled
                    continue
                # true samples ---------------------------
                # nts_tr: (batch, in_seq_len)
                # tms_tr: (batch, in_seq_len, 3)
                nts_tr, tms_tr = true_samples[:, :, 0], true_samples[:, :, 1:]
                # random vectors ---------------------------
                # nt_ltnt: (batch, out_seq_len, 16)
                # tm_ltnt: (batch, out_seq_len, 1)
                nt_ltnt = util.latant_vector(self.batch_size, self.in_seq_len, 16, mean_=0.0, std_=1.1)
                tm_ltnt = util.latant_vector(self.batch_size, self.in_seq_len, 1, mean_=0.0, std_=1.1)

                losses = self.train_step(nt_ltnt, tm_ltnt, nts_tr, tms_tr, freeze_disc=True)




                nt_ltnt2 = util.latant_vector(self.batch_size, self.in_seq_len, 16, mean_=0.0, std_=1.1)
                tm_ltnt2 = util.latant_vector(self.batch_size, self.in_seq_len, 1, mean_=0.0, std_=1.1)



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
