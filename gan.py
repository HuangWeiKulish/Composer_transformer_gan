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

notes_latent_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_latent'
time_latent_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_latent'

notes_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_embedder'
notes_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_generator'
time_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_generator'

notes_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_discriminator'
time_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_discriminator'
combine_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/discriminator'

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

                 mode_='both'):
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

        self.optimizer_disc = None
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss')
        self.optimizer_gen = None
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss')

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
        # todo: generate music from latent

        pass

    def load_model(self, notes_latent_path=notes_latent_path, time_latent_path=time_latent_path,
                   notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path, time_gen_path=time_gen_path,
                   notes_disc_path=notes_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
                   load_notes_ltnt=True, load_time_ltnt=True, load_notes_emb=True,
                   load_notes_gen=True, load_time_gen=True, load_disc=True,
                   train_ntlatent=True, train_tmlatent=True, train_ntemb=True,
                   train_ntgen=True, train_tmgen=True, train_disc=True, max_to_keep=5):

        # ---------------------- call back setting --------------------------
        # load latent models
        if self.mode_ in ['notes', 'both']:
            if load_notes_ltnt:
                self.cp_notes_ltnt = tf.train.Checkpoint(model=self.notes_latent, optimizer=self.optimizer)
                self.cp_manager_notes_ltnt = tf.train.CheckpointManager(
                    self.cp_notes_ltnt, notes_latent_path, max_to_keep=max_to_keep)
                if self.cp_manager_notes_ltnt.latest_checkpoint:
                    self.cp_notes_ltnt.restore(self.cp_manager_notes_ltnt.latest_checkpoint)
                    print('Restored the latest notes_ltnt')
        if self.mode_ in ['time', 'both']:
            if load_time_ltnt:
                self.cp_time_ltnt = tf.train.Checkpoint(model=self.time_latent, optimizer=self.optimizer)
                self.cp_manager_time_ltnt = tf.train.CheckpointManager(
                    self.cp_time_ltnt, time_latent_path, max_to_keep=max_to_keep)
                if self.cp_manager_time_ltnt.latest_checkpoint:
                    self.cp_time_ltnt.restore(self.cp_manager_time_ltnt.latest_checkpoint)
                    print('Restored the latest time_ltnt')

        # load notes embedder and generator
        if self.mode_ in ['notes', 'both']:
            if load_notes_emb:
                self.cp_notes_emb = tf.train.Checkpoint(model=self.gen.notes_emb, optimizer=self.optimizer)
                self.cp_manager_notes_emb = tf.train.CheckpointManager(
                    self.cp_notes_emb, notes_emb_path, max_to_keep=max_to_keep)
                if self.cp_manager_notes_emb.latest_checkpoint:
                    self.cp_notes_emb.restore(self.cp_manager_notes_emb.latest_checkpoint)
                    print('Restored the latest notes_emb')
            if load_notes_gen:
                self.cp_notes_gen = tf.train.Checkpoint(model=self.gen.notes_gen, optimizer=self.optimizer)
                self.cp_manager_notes_gen = tf.train.CheckpointManager(
                    self.cp_notes_gen, notes_gen_path, max_to_keep=max_to_keep)
                if self.cp_manager_notes_gen.latest_checkpoint:
                    self.cp_notes_gen.restore(self.cp_manager_notes_gen.latest_checkpoint)
                    print('Restored the latest notes_gen')

        if self.mode_ in ['time', 'both']:
            if load_time_gen:
                self.cp_time_gen = tf.train.Checkpoint(model=self.gen.time_gen, optimizer=self.optimizer)
                self.cp_manager_time_gen = tf.train.CheckpointManager(
                    self.cp_time_gen, time_gen_path, max_to_keep=max_to_keep)
                if self.cp_manager_time_gen.latest_checkpoint:
                    self.cp_time_gen.restore(self.cp_manager_time_gen.latest_checkpoint)
                    print('Restored the latest time_gen')

        # load discriminator
        if self.mode_ == 'notes':
            disc_pth = notes_disc_path
        elif self.mode_ == 'time':
            disc_pth = time_disc_path
        else:  # self.mode_ == 'both'
            disc_pth = combine_disc_path
        if load_disc:
            self.cp_disc = tf.train.Checkpoint(model=self.disc, optimizer=self.optimizer)
            self.cp_manager_disc = tf.train.CheckpointManager(self.cp_disc, disc_pth, max_to_keep=max_to_keep)
            if self.cp_manager_disc.latest_checkpoint:
                self.cp_disc.restore(self.cp_manager_disc.latest_checkpoint)
                print('Restored the latest discriminator for {}'.format(self.mode_))

        # ---------------------- set trainable --------------------------
        self.train_ntlatent = train_ntlatent
        self.train_tmlatent = train_tmlatent
        self.train_ntemb = train_ntemb
        self.train_ntgen = train_ntgen
        self.train_tmgen = train_tmgen
        self.train_disc = train_disc

        if train_ntlatent:
            util.model_trainable(self.notes_latent, trainable=train_ntlatent)
        if train_tmlatent:
            util.model_trainable(self.time_latent, trainable=train_tmlatent)
        if train_ntemb & (self.mode_ in ['notes', 'both']):
            util.model_trainable(self.gen.notes_emb, trainable=train_ntemb)
        if train_ntgen & (self.mode_ in ['notes', 'both']):
            util.model_trainable(self.gen.notes_gen, trainable=train_ntgen)
        if train_tmgen & (self.mode_ in ['time', 'both']):
            util.model_trainable(self.gen.time_gen, trainable=train_tmgen)
        if train_disc:
            util.model_trainable(self.disc, trainable=train_disc)

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

        # unfreeze discriminator ------------------------------------------------------------------
        if self.train_disc:
            util.model_trainable(self.disc, trainable=True)

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

        # discriminator loss ------------------------------------------------------------------
        # lbl_comb: (batch, 1)
        # pred_comb: (batch, 1)
        pred_comb = self.disc(nts_comb, tms_comb, de_in)
        loss_disc = tf.keras.losses.binary_crossentropy(lbl_comb, pred_comb, from_logits=False, label_smoothing=0)
        return loss_disc, self.disc.trainable_variables

    def train_generator(self, nt_ltnt, tm_ltnt):
        # nt_ltnt: (batch, in_seq_len, 16)
        # tm_ltnt: (batch, in_seq_len, 1)

        # freeze discriminator ------------------------------------------------------------------
        if self.train_disc:
            util.model_trainable(self.disc, trainable=False)

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
            # todo: return the fake sample as well

        # prepare decode initial input ------------------------------------------------------------------
        # de_in: (batch, 1, embed_dim)
        de_in = self.gen.notes_emb(tf.constant([[0]] * self.batch_size, dtype=tf.float32)) \
            if self.mode_ == 'time' \
            else self.gen.notes_emb(tf.constant([[self.strt_token_id]] * self.batch_size, dtype=tf.float32))

        # combine samples ------------------------------------------------------------------
        # lbl_fk: (bacth, 1)
        lbl_fk = tf.constant([[1]] * self.batch_size, dtype=tf.float32)

        # generator loss ------------------------------------------------------------------
        pred_fk = self.disc(nts_fk, tms_fk, de_in)
        loss_gen = tf.keras.losses.binary_crossentropy(lbl_fk, pred_fk, from_logits=False, label_smoothing=0)

        variables_gen = self.trainable_variables
        if self.mode_ in ['notes', 'both']:
            variables_gen += self.notes_latent.trainable_variables
        if self.mode_ in ['time', 'both']:
            variables_gen += self.time_latent.trainable_variables

        return loss_gen, variables_gen

    def train_step(self, nt_ltnt, tm_ltnt, nt_ltnt2, tm_ltnt2, nts_tr, tms_tr):
        # nt_ltnt: notes random vector (batch, out_seq_len, 16)
        # tm_ltnt: time random vector (batch, out_seq_len, 1)
        # nt_ltnt2: notes random vector (batch, out_seq_len, 16)
        # tm_ltnt2: time random vector (batch, out_seq_len, 1)
        # nts_tr: true sample notes (batch, in_seq_len)
        # tms_tr: true sample time (batch, in_seq_len, 3)

        with tf.GradientTape() as tape:
            # Step 1. train discriminator with combined true and fake samples;
            loss_disc, variables_disc = self.train_discriminator(nt_ltnt, tm_ltnt, nts_tr, tms_tr)
            loss_disc_fake, loss_disc_true = loss_disc[:self.batch_size], loss_disc[self.batch_size:]

            gradients_disc = tape.gradient(loss_disc, variables_disc)
            self.optimizer_disc.apply_gradients(zip(gradients_disc, variables_disc))
            self.train_loss_disc(loss_disc)

            # Step 2: freeze discriminator and use the fake sample with true label to train generator
            loss_gen, variables_gen = self.train_generator(nt_ltnt2, tm_ltnt2)
            gradients_gen = tape.gradient(loss_gen, variables_gen)
            self.optimizer_gen.apply_gradients(zip(gradients_gen, variables_gen))
            self.train_loss_gen(loss_gen)

            # Todo: if trainable variables is empty list, will it affect gradient calculation?????
        return loss_disc_fake, loss_disc_true, loss_disc, loss_gen

    def train(self, epochs=10, save_model_step=1, save_sample_step=5,
              print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=5,
              lr_gen=0.01, lr_disc=0.0001, warmup_steps=4000, custm_lr=True,
              optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
              notes_latent_path=notes_latent_path, time_latent_path=time_latent_path,
              notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path, time_gen_path=time_gen_path,
              notes_disc_path=notes_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
              load_notes_ltnt=True, load_time_ltnt=True, load_notes_emb=True,
              load_notes_gen=True, load_time_gen=True, load_disc=True,
              train_ntlatent=True, train_tmlatent=True, train_ntemb=True,
              train_ntgen=True, train_tmgen=True, train_disc=True,
              save_notes_ltnt=True, save_time_ltnt=True, save_notes_emb=True,
              save_notes_gen=True, save_time_gen=True, save_disc=True,
              max_to_keep=5):

        self.load_model(
            notes_latent_path=notes_latent_path, time_latent_path=time_latent_path,
            notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path, time_gen_path=time_gen_path,
            notes_disc_path=notes_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
            load_notes_ltnt=load_notes_ltnt, load_time_ltnt=load_time_ltnt, load_notes_emb=load_notes_emb,
            load_notes_gen=load_notes_gen, load_time_gen=load_time_gen, load_disc=load_disc,
            train_ntlatent=train_ntlatent, train_tmlatent=train_tmlatent, train_ntemb=train_ntemb,
            train_ntgen=train_ntgen, train_tmgen=train_tmgen, train_disc=train_disc, max_to_keep=max_to_keep)

        if custm_lr:
            lr_gen = util.CustomSchedule(self.embed_dim, warmup_steps)
            lr_disc = util.CustomSchedule(self.embed_dim, warmup_steps)
        self.optimizer_disc = optmzr(lr_disc)
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')
        self.optimizer_gen = optmzr(lr_gen)
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')

        # ---------------------- training --------------------------
        for epoch in range(epochs):
            self.train_loss_disc.reset_states()
            self.train_loss_gen.reset_states()

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
                # nt_ltnt2: (batch, out_seq_len, 16)
                # tm_ltnt2: (batch, out_seq_len, 1)
                nt_ltnt2 = util.latant_vector(self.batch_size, self.in_seq_len, 16, mean_=0.0, std_=1.1)
                tm_ltnt2 = util.latant_vector(self.batch_size, self.in_seq_len, 1, mean_=0.0, std_=1.1)

                loss_disc_fake, loss_disc_true, loss_disc, loss_gen = self.train_step(
                    nt_ltnt, tm_ltnt, nt_ltnt2, tm_ltnt2, nts_tr, tms_tr)

                if print_batch:
                    if (i + 1) % print_batch_step == 0:
                        print('Epoch {} Batch {}: gen_loss={:.4f}; '
                              'disc_loss={:.4f} (fake_loss={}, true_loss={});'.format(
                            epoch+1, i+1, loss_gen.numpy(),
                            loss_disc.numpy(), loss_disc_fake.numpy(), loss_disc_true.numpy()))

            if print_epoch:
                if (epoch + 1) % print_epoch_step == 0:
                    print('Epoch {}: Loss_gen={:.4f}, Loss_disc={:.4f}, Time used={:.4f}'.format(
                        epoch+1, self.train_loss_gen.result(), self.train_loss_disc.result(), time.time()-start))

            if (epoch+1) % save_model_step == 0:
                if self.mode_ in ['notes', 'both']:
                    if save_notes_ltnt:
                        self.cp_manager_notes_ltnt.save()
                        print('Saved the latest notes_ltnt')
                    if save_notes_emb:
                        self.cp_manager_notes_emb.save()
                        print('Saved the latest notes_emb')
                    if save_notes_gen:
                        self.cp_manager_notes_gen.save()
                        print('Saved the latest notes_gen')

                if self.mode_ in ['time', 'both']:
                    if save_time_ltnt:
                        self.cp_manager_time_ltnt.save()
                        print('Saved the latest time_ltnt')
                    if save_time_gen:
                        self.cp_manager_time_gen.save()
                        print('Saved the latest time_gen')
                if save_disc:
                    self.cp_manager_disc.save()
                    print('Saved the latest discriminator for {}'.format(self.mode_))

            if save_sample_step:
                # todo: save midi file from randomly sampled
                print(save_sample_step)


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
train_ntlatent=True
train_tmlatent=True
train_ntemb=True
train_ntgen=True
train_tmgen=True
train_disc=True
lr_tm=0.01
warmup_steps=4000
optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
strt_token_id=15001  # tk.word_index['<start>']
"""

