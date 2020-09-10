import os
import glob
import json
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pickle as pkl
import tensorflow as tf
import util
import discriminator
import generator
import preprocess
import transformer

notes_latent_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_latent'
time_latent_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_latent'

notes_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_embedder'
notes_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_generator'
time_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_generator'

notes_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_discriminator'
time_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_discriminator'
combine_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/discriminator'

result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'

# tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_indexcer/notes_dict_final.pkl'
# tk = pkl.load(open(tk_path, 'rb'))


class GAN(tf.keras.Model):

    def __init__(self, in_seq_len=16, out_seq_len=64, strt_token_id=15001, out_notes_pool_size=15002, embed_dim=256,
                 n_heads=4, max_pos=800, time_features=3, fc_activation="relu",
                 g_encoder_layers=2, g_decoder_layers=2, g_fc_layers=3, g_norm_epsilon=1e-6,
                 g_embedding_dropout_rate=0.2, g_transformer_dropout_rate=0.2,
                 d_kernel_size=3, d_encoder_layers=2, d_decoder_layers=2, d_fc_layers=3, d_norm_epsilon=1e-6,
                 d_transformer_dropout_rate=0.2,
                 notes_latent_nlayers=4, notes_latent_dim_base=4, time_latent_nlayers=4,
                 mode_='both'):
        super(GAN, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'

        # ---------------------------------- settings ----------------------------------
        self.batch_size = None
        self.true_data = None
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.strt_token_id = strt_token_id  # strt_token_id = tk.word_index['<start>']
        self.mode_ = mode_  # only choose from ['notes', 'time', 'both']
        self.embed_dim = embed_dim

        self.optimizer_disc = None
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')
        self.optimizer_gen = None
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')

        # ---------------------------------- layers ----------------------------------
        # latent vector generator
        self.notes_latent = generator.NotesLatent(nlayers=notes_latent_nlayers, dim_base=notes_latent_dim_base)
        self.time_latent = generator.TimeLatent(nlayers=time_latent_nlayers)

        # notes generator
        self.notes_emb = generator.NotesEmbedder(
            notes_pool_size=out_notes_pool_size, max_pos=max_pos, embed_dim=embed_dim,
            dropout_rate=g_embedding_dropout_rate)  # todo: why self.notes_emb works but self.notes_latent doesn't work??
        self.notes_gen = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=out_notes_pool_size,
            encoder_layers=g_encoder_layers, decoder_layers=g_decoder_layers, fc_layers=g_fc_layers,
            norm_epsilon=g_norm_epsilon, dropout_rate=g_transformer_dropout_rate, fc_activation=fc_activation,
            out_positive=False)

        # time generator
        # 3 for [velocity, velocity, time since last start, notes duration]
        self.time_gen = transformer.Transformer(
            embed_dim=time_features, n_heads=1, out_notes_pool_size=time_features,
            encoder_layers=g_encoder_layers, decoder_layers=g_decoder_layers, fc_layers=g_fc_layers,
            norm_epsilon=g_norm_epsilon, dropout_rate=g_transformer_dropout_rate, fc_activation=fc_activation,
            out_positive=True)

        # discriminator
        if self.mode_ == 'notes':
            self.disc = discriminator.NotesDiscriminator(
                embed_dim=embed_dim, n_heads=n_heads, fc_activation=fc_activation, encoder_layers=d_encoder_layers,
                decoder_layers=d_decoder_layers, fc_layers=d_fc_layers, norm_epsilon=d_norm_epsilon,
                transformer_dropout_rate=d_transformer_dropout_rate)
        elif self.mode_ == 'time':
            self.disc = discriminator.TimeDiscriminator(
                time_features=time_features, fc_activation=fc_activation, encoder_layers=d_encoder_layers,
                decoder_layers=d_decoder_layers, fc_layers=d_fc_layers, norm_epsilon=d_norm_epsilon,
                transformer_dropout_rate=d_transformer_dropout_rate)
        else:  # self.mode_ == 'both'
            self.disc = discriminator.Discriminator(
                embed_dim=embed_dim, n_heads=n_heads, kernel_size=d_kernel_size, fc_activation=fc_activation,
                encoder_layers=d_encoder_layers, decoder_layers=d_decoder_layers, fc_layers=d_fc_layers,
                norm_epsilon=d_norm_epsilon, transformer_dropout_rate=d_transformer_dropout_rate)

    def load_true_samples(self, tk, step=30, batch_size=10, vel_norm=64.0, tmps_norm=0.12,
                          dur_norm=1.3, pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['']):
        self.vel_norm = vel_norm
        self.tmps_norm = tmps_norm
        self.dur_norm = dur_norm
        self.batch_size = batch_size
        self.tk = tk
        self.true_data = util.load_true_data_gan(
            tk, self.out_seq_len, step=step, batch_size=batch_size, vel_norm=vel_norm,
            tmps_norm=tmps_norm, dur_norm=dur_norm, pths=pths, name_substr_list=name_substr_list)

    def load_model(self, notes_latent_path=notes_latent_path, time_latent_path=time_latent_path,
                   notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path, time_gen_path=time_gen_path,
                   notes_disc_path=notes_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
                   train_ntlatent=True, train_tmlatent=True, train_ntemb=True,
                   train_ntgen=True, train_tmgen=True, train_disc=True, max_to_keep=5):

        # ---------------------- call back setting --------------------------
        # load latent models
        self.cp_notes_ltnt = tf.train.Checkpoint(model=self.notes_latent, optimizer=self.optimizer_gen)
        self.cp_manager_notes_ltnt = tf.train.CheckpointManager(
            self.cp_notes_ltnt, notes_latent_path, max_to_keep=max_to_keep)
        if self.cp_manager_notes_ltnt.latest_checkpoint:
            self.cp_notes_ltnt.restore(self.cp_manager_notes_ltnt.latest_checkpoint)
            print('Restored the latest notes_ltnt')

        self.cp_time_ltnt = tf.train.Checkpoint(model=self.time_latent, optimizer=self.optimizer_gen)
        self.cp_manager_time_ltnt = tf.train.CheckpointManager(
            self.cp_time_ltnt, time_latent_path, max_to_keep=max_to_keep)
        if self.cp_manager_time_ltnt.latest_checkpoint:
            self.cp_time_ltnt.restore(self.cp_manager_time_ltnt.latest_checkpoint)
            print('Restored the latest time_ltnt')

        # load generators
        self.cp_notes_emb = tf.train.Checkpoint(model=self.notes_emb, optimizer=self.optimizer_gen)
        self.cp_manager_notes_emb = tf.train.CheckpointManager(
            self.cp_notes_emb, notes_emb_path, max_to_keep=max_to_keep)
        if self.cp_manager_notes_emb.latest_checkpoint:
            self.cp_notes_emb.restore(self.cp_manager_notes_emb.latest_checkpoint)
            print('Restored the latest notes_emb')

        self.cp_notes_gen = tf.train.Checkpoint(model=self.notes_gen, optimizer=self.optimizer_gen)
        self.cp_manager_notes_gen = tf.train.CheckpointManager(
            self.cp_notes_gen, notes_gen_path, max_to_keep=max_to_keep)
        if self.cp_manager_notes_gen.latest_checkpoint:
            self.cp_notes_gen.restore(self.cp_manager_notes_gen.latest_checkpoint)
            print('Restored the latest notes_gen')

        self.cp_time_gen = tf.train.Checkpoint(model=self.time_gen, optimizer=self.optimizer_gen)
        self.cp_manager_time_gen = tf.train.CheckpointManager(
            self.cp_time_gen, time_gen_path, max_to_keep=max_to_keep)
        if self.cp_manager_time_gen.latest_checkpoint:
            self.cp_time_gen.restore(self.cp_manager_time_gen.latest_checkpoint)
            print('Restored the latest time_gen')

        # load discriminator
        if self.mode_ == 'notes':
            disc_pth = notes_disc_path
            str_ = 'Restored the latest notes_disc'
        elif self.mode_ == 'time':
            disc_pth = time_disc_path
            str_ = 'Restored the latest time_disc'
        else:  # self.mode_ == 'both'
            disc_pth = combine_disc_path
            str_ = 'Restored the latest combine_disc'
        self.cp_disc = tf.train.Checkpoint(model=self.disc, optimizer=self.optimizer_disc)
        self.cp_manager_disc = tf.train.CheckpointManager(self.cp_disc, disc_pth, max_to_keep=max_to_keep)
        if self.cp_manager_disc.latest_checkpoint:
            self.cp_disc.restore(self.cp_manager_disc.latest_checkpoint)
            print(str_)

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
            util.model_trainable(self.notes_emb, trainable=train_ntemb)
        if train_ntgen & (self.mode_ in ['notes', 'both']):
            util.model_trainable(self.notes_gen, trainable=train_ntgen)
        if train_tmgen & (self.mode_ in ['time', 'both']):
            util.model_trainable(self.time_gen, trainable=train_tmgen)
        if train_disc:
            util.model_trainable(self.disc, trainable=train_disc)

    def predict_notes(self, x_en_nt, return_str=False):
        # x_en: (batch_size, in_seq_len, embed_dim)
        # x_de: (batch_size, 1, embed_dim)
        batch_size = x_en_nt.shape[0]
        x_de = tf.constant([[self.tk.word_index['<start>']]] * batch_size, dtype=tf.float32)
        x_de = self.notes_emb(x_de)
        result = []  # (out_seq_len, batch)
        for i in range(self.out_seq_len):
            mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
            # x_out: (batch_size, 1, out_notes_pool_size)
            x_out, _ = self.notes_gen(x_en=x_en_nt, x_de=x_de, mask_padding=mask_padding, mask_lookahead=mask_lookahead)
            # translate prediction to text
            pred = tf.argmax(x_out, -1)
            pred = [nid for nid in pred.numpy()[:, -1]]  # len = batch, take the last prediction
            # append pred to x_de:
            x_de = tf.concat((x_de, self.notes_emb(np.expand_dims(pred, axis=-1))), axis=1)
            if return_str:
                # !! use pd + 1 because tk index starts from 1
                result.append([self.tk.index_word[pd + 1] for pd in pred])  # return notes string
            else:
                result.append(pred)  # return notes index
        result = np.transpose(np.array(result, dtype=object), (1, 0))  # (batch, out_seq_len)
        return result  # notes string

    def predict_time(self, x_en_tm, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3, return_denorm=False):
        # x_en: (batch_size, in_seq_len, 3)
        batch_size = x_en_tm.shape[0]
        # init x_de: (batch_size, 1, 3)
        x_de = tf.constant([[[0] * 3]] * batch_size, dtype=tf.float32)
        result = []  # (out_seq_len, batch, 3)
        for i in range(self.out_seq_len):
            mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
            # x_out: (batch_size, 1, 3)
            x_out, _ = self.time_gen(x_en=x_en_tm, x_de=x_de, mask_padding=mask_padding, mask_lookahead=mask_lookahead)
            pred = x_out[:, -1, :][:, tf.newaxis, :]  # only take the last prediction
            x_de = tf.concat((x_de, pred), axis=1)
            result.append(x_out[:, -1, :].numpy().tolist())  # only take the last prediction
        result = np.transpose(np.array(result, dtype=object), (1, 0, 2))  # (batch, out_seq_len, 3)
        if return_denorm:
            result = result * np.array([vel_norm, tmps_norm, dur_norm])
        return result

    def prepare_fake_notes(self, nt_ltnt, return_str=False):
        # create latent vectors from random inputs ----------------------
        # (batch, in_seq_len, embed_dim)
        nt_ltnt_ = self.notes_latent(nt_ltnt)
        # generate music from latent vectors ----------------------
        # get nts: (batch, out_seq_len)
        nts = self.predict_notes(nt_ltnt_, return_str=return_str)
        nts = tf.convert_to_tensor(nts, dtype=tf.float32)  # convert from numpy to tensor
        nts = self.notes_emb(nts)
        return nts

    def prepare_fake_time(self, tm_ltnt, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3, return_denorm=False):
        # create latent vectors from random inputs ----------------------
        # (batch, in_seq_len, 3)
        tm_ltnt_ = self.time_latent(tm_ltnt) if self.mode_ in ['time', 'both'] else None
        # generate music from latent vectors ----------------------
        # get tms: (batch, out_seq_len, 3)
        tms = self.predict_time(tm_ltnt_, vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
                                return_denorm=return_denorm)
        tms = tf.convert_to_tensor(tms, dtype=tf.float32)  # convert from numpy to tensor
        return tms

    def train_discriminator(self, nt_ltnt, tm_ltnt, nts_tr, tms_tr):
        # nt_ltnt: (batch, in_seq_len, 16) np.array
        # tm_ltnt: (batch, in_seq_len, 1) np.array
        # nts_tr: (batch, out_seq_len) tf.tensor
        # tms_tr: (batch, out_seq_len, 3) tf.tensor

        # unfreeze discriminator ------------------------------------------------------------------
        if self.train_disc:
            util.model_trainable(self.disc, trainable=True)

        # prepare fake samples ------------------------------------------------------------------
        # nts_fk: (batch, out_seq_len, embed_dim) or None
        # tms_fk: (batch, out_seq_len, 3) or None
        nts_fk, tms_fk = self.prepare_fake_samples(
            nt_ltnt, tm_ltnt, return_str=False, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3, return_denorm=False)

        # prepare true samples ------------------------------------------------------------------
        # nts_tr: (batch, out_seq_len, embed_dim)
        if self.mode_ in ['notes', 'both']:
            nts_tr = self.notes_emb(nts_tr)

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
        if self.mode_ == 'notes':
            # de_in: (batch * 2, 1, embed_dim)
            de_in = self.notes_emb(tf.constant([[self.strt_token_id]] * self.batch_size * 2, dtype=tf.float32))
            pred_comb = self.disc(nts_comb, de_in)
        elif self.mode_ == 'time':
            # de_in: (batch * 2, 1, 3)
            de_in = tf.constant([[[0] * 3]] * self.batch_size * 2, dtype=tf.float32)
            pred_comb = self.disc(tms_comb, de_in)
        else:  # self.mode_ == 'both'
            # de_in: (batch * 2, 1, embed_dim)
            de_in = self.notes_emb(tf.constant([[self.strt_token_id]] * self.batch_size * 2, dtype=tf.float32))
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
        nts_fk = self.prepare_fake_notes(nt_ltnt, return_str=False)
        tms_fk = self.prepare_fake_time(tm_ltnt, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3, return_denorm=False)

        # combine samples ------------------------------------------------------------------
        # lbl_fk: (bacth, 1)
        lbl_fk = tf.constant([[1]] * self.batch_size, dtype=tf.float32)

        # generator loss ------------------------------------------------------------------
        if self.mode_ == 'notes':
            # de_in: (batch, 1, embed_dim)
            de_in = self.notes_emb(tf.constant([[self.strt_token_id]] * self.batch_size, dtype=tf.float32))
            pred_fk = self.disc(nts_fk, de_in)
        elif self.mode_ == 'time':
            # de_in: (batch, 1, 3)
            de_in = tf.constant([[[0] * 3]] * self.batch_size, dtype=tf.float32)
            pred_fk = self.disc(tms_fk, de_in)
        else:  # self.mode_ == 'both'
            # de_in: (batch, 1, embed_dim)
            de_in = self.notes_emb(tf.constant([[self.strt_token_id]] * self.batch_size, dtype=tf.float32))
            pred_fk = self.disc(nts_fk, tms_fk, de_in)

        loss_gen = tf.keras.losses.binary_crossentropy(lbl_fk, pred_fk, from_logits=False, label_smoothing=0)
        variables_gen = self.notes_latent.trainable_variables if self.mode_ in ['notes', 'both'] else [] + \
                        self.time_latent.trainable_variables if self.mode_ in ['time', 'both'] else [] + \
                        self.notes_emb.trainable_variables + \
                        self.notes_gen.trainable_variables if self.mode_ in ['notes', 'both'] else [] + \
                        self.time_gen.trainable_variables if self.mode_ in ['time', 'both'] else []
        print('loss_gen', loss_gen)
        print('variables_gen', len(variables_gen))

        return loss_gen, variables_gen

    def train_step(self, nt_ltnt, tm_ltnt, nt_ltnt2, tm_ltnt2, nts_tr, tms_tr):
        # nt_ltnt: notes random vector (batch, out_seq_len, 16)
        # tm_ltnt: time random vector (batch, out_seq_len, 1)
        # nt_ltnt2: notes random vector (batch, out_seq_len, 16)
        # tm_ltnt2: time random vector (batch, out_seq_len, 1)
        # nts_tr: true sample notes (batch, in_seq_len)
        # tms_tr: true sample time (batch, in_seq_len, 3)

        # Step 1. train discriminator with combined true and fake samples; --------------------
        with tf.GradientTape() as tape:
            loss_disc, variables_disc = self.train_discriminator(nt_ltnt, tm_ltnt, nts_tr, tms_tr)
            loss_disc_fake, loss_disc_true = loss_disc[:self.batch_size], loss_disc[self.batch_size:]
            gradients_disc = tape.gradient(loss_disc, variables_disc)
            self.optimizer_disc.apply_gradients(zip(gradients_disc, variables_disc))
            self.train_loss_disc(loss_disc)

        # Step 2: freeze discriminator and use the fake sample with true label to train generator ---------------
        with tf.GradientTape() as tape:
            loss_gen, variables_gen = self.train_generator(nt_ltnt2, tm_ltnt2)
            gradients_gen = tape.gradient(loss_gen, variables_gen)
            self.optimizer_gen.apply_gradients(zip(gradients_gen, variables_gen))  # todo: ValueError: No gradients provided for any variable
            self.train_loss_gen(loss_gen)
            print('train generator success')

            # Todo: if trainable variables is empty list, will it affect gradient calculation?????
        return loss_disc_fake, loss_disc_true, loss_disc, loss_gen

    def train(self, epochs=10, save_model_step=1, save_sample_step=1,
              print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=5,
              lr_gen=0.01, lr_disc=0.0001, warmup_steps=4000,
              optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
              notes_latent_path=notes_latent_path, time_latent_path=time_latent_path,
              notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path, time_gen_path=time_gen_path,
              notes_disc_path=notes_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
              result_path=result_path,
              train_ntlatent=True, train_tmlatent=True, train_ntemb=True,
              train_ntgen=True, train_tmgen=True, train_disc=True,
              save_notes_ltnt=True, save_time_ltnt=True, save_notes_emb=True,
              save_notes_gen=True, save_time_gen=True, save_disc=True,
              max_to_keep=5):

        log_current = {'epochs': 1, 'lr_gen': lr_gen, 'lr_disc': lr_disc, 'warmup_steps': warmup_steps,
                       'mode': self.mode_}
        try:
            log = json.load(open(os.path.join(result_path, 'log.json'), 'r'))
            log_current['epochs'] = log[-1]['epochs']
        except:
            log = []
        last_ep = log_current['epochs']


        lr_gen = util.CustomSchedule(self.embed_dim, warmup_steps)
        lr_disc = util.CustomSchedule(self.embed_dim, warmup_steps)
        self.optimizer_disc = optmzr(lr_disc)
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')
        self.optimizer_gen = optmzr(lr_gen)
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')

        self.train_ntlatent = train_ntlatent
        self.train_tmlatent = train_tmlatent
        self.train_ntemb = train_ntemb
        self.train_ntgen = train_ntgen
        self.train_tmgen = train_tmgen
        self.train_disc = train_disc
        self.load_model(
            notes_latent_path=notes_latent_path, time_latent_path=time_latent_path,
            notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path, time_gen_path=time_gen_path,
            notes_disc_path=notes_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
            train_ntlatent=train_ntlatent, train_tmlatent=train_tmlatent, train_ntemb=train_ntemb,
            train_ntgen=train_ntgen, train_tmgen=train_tmgen, train_disc=train_disc, max_to_keep=max_to_keep)

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

                log_current['epochs'] += epoch
                log.append(log_current)
                json.dump(log, open(os.path.join(result_path, 'log.json'), 'w'))

            if (epoch+1) % save_sample_step == 0:
                mid = self.generate_music()
                file_name = os.path.join(result_path, self.mode_, 'ep{}.mid'.format(last_ep))
                mid.save(file_name)
                print('Saved a fake sample: {}'.format(file_name))

            last_ep += 1

    def generate_music(self, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3):
        nt_ltnt = util.latant_vector(1, self.in_seq_len, 16, mean_=0.0, std_=1.1)  # (1, 16, 16)
        tm_ltnt = util.latant_vector(1, self.in_seq_len, 1, mean_=0.0, std_=1.1)  # (1, 16, 1)
        nts, tms = self.prepare_fake_samples(
            nt_ltnt, tm_ltnt, return_str=True, vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
            return_denorm=True)
        tms = np.array([[self.vel_norm, self.tmps_norm, self.dur_norm]] * self.out_seq_len)[np.newaxis, :, :] \
            if tms is None else tms
        nts = np.array([['64'] * self.out_seq_len]) if nts is None else nts
        # nts: (1, out_seq_len), string
        # tms: (1, out_seq_len, 3)
        ary = np.squeeze(np.concatenate([nts[:, :, np.newaxis], abs(tms)], axis=-1), axis=0)  # (out_seq_len, 4)
        mid = preprocess.Conversion.arry2mid(ary)
        return mid





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

