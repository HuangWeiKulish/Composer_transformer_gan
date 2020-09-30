import os
import time
import numpy as np
import tensorflow as tf
import util
import discriminator
import generator
import preprocess
import gensim

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

path_base = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/'
model_paths = {k: os.path.join(path_base, k) for k in
               ['chords_embedder', 'chords_style', 'chords_syn', 'time_style', 'time_syn',
                'chords_disc', 'time_disc', 'comb_disc']}
result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'


class WGAN(tf.keras.models.Model):

    def __init__(self,
                 in_dim=512, embed_dim=16, latent_std=1.0, strt_dim=3, n_heads=4, init_knl=3,
                 max_pos=1000, pos_conv_knl=3,
                 chstl_fc_layers=4, chstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1),
                 chsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                 chsyn_encoder_layers=3, chsyn_decoder_layers=3, chsyn_fc_layers=3, chsyn_norm_epsilon=1e-6,
                 chsyn_transformer_dropout_rate=0.2, chsyn_noise_std=0.5,
                 time_features=3, tmstl_fc_layers=4, tmstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1),
                 tmsyn_encoder_layers=3, tmsyn_decoder_layers=3, tmsyn_fc_layers=3, tmsyn_norm_epsilon=1e-6,
                 tmsyn_transformer_dropout_rate=0.2, tmsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                 tmsyn_noise_std=0.5,
                 d_kernel_size=3, d_encoder_layers=1, d_decoder_layers=1, d_fc_layers=3, d_norm_epsilon=1e-6,
                 d_transformer_dropout_rate=0.2, d_fc_activation=tf.keras.activations.tanh,
                 d_out_dropout=0.3, d_recycle_fc_activ=tf.keras.activations.elu, mode_='comb'):
        super(WGAN, self).__init__()
        if mode_ in ['chords', 'comb']:
            assert embed_dim % n_heads == 0, 'make sure: embed_dim % chsyn_n_heads == 0'

        # ---------------------------------- settings ----------------------------------
        self.mode_ = mode_
        self.embed_dim = embed_dim
        self.time_features = time_features  # 3 for [velocity, velocity, time since last start, chords duration]
        self.in_dim = in_dim
        self.strt_dim = strt_dim
        self.latent_std = latent_std
        # callback settings
        self.ckpts = dict()
        self.ckpt_managers = dict()
        # optimisers
        self.optimizer_gen = tf.keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.optimizer_disc = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # losses
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')

        # ---------------------------------- layers ----------------------------------
        # generators
        if mode_ != 'time':
            self.chords_style = generator.Mapping(fc_layers=chstl_fc_layers, activ=chstl_activ)
            self.chords_syn = generator.ChordsSynthesis(
                embed_dim=embed_dim, init_knl=init_knl, strt_dim=strt_dim,
                n_heads=n_heads, fc_activation=chsyn_fc_activation, encoder_layers=chsyn_encoder_layers,
                decoder_layers=chsyn_decoder_layers, fc_layers=chsyn_fc_layers, norm_epsilon=chsyn_norm_epsilon,
                transformer_dropout_rate=chsyn_transformer_dropout_rate, noise_std=chsyn_noise_std)
        if mode_ != 'chords':
            self.time_style = generator.Mapping(fc_layers=tmstl_fc_layers, activ=tmstl_activ)
            self.time_syn = generator.TimeSynthesis(
                time_features=time_features, init_knl=init_knl, strt_dim=strt_dim, fc_activation=tmsyn_fc_activation,
                encoder_layers=tmsyn_encoder_layers, decoder_layers=tmsyn_decoder_layers, fc_layers=tmsyn_fc_layers,
                norm_epsilon=tmsyn_norm_epsilon, transformer_dropout_rate=tmsyn_transformer_dropout_rate,
                noise_std=tmsyn_noise_std, max_pos=max_pos, pos_conv_knl=pos_conv_knl)

        # discriminator
        if mode_ == 'chords':
            self.disc = discriminator.ChordsDiscriminator(
                embed_dim=embed_dim, n_heads=n_heads, fc_activation=d_fc_activation,
                encoder_layers=d_encoder_layers, decoder_layers=d_decoder_layers, fc_layers=d_fc_layers,
                norm_epsilon=d_norm_epsilon, transformer_dropout_rate=d_transformer_dropout_rate,
                pre_out_dim=in_dim, out_dropout=d_out_dropout, recycle_fc_activ=d_recycle_fc_activ)
        elif mode_ == 'time':
            self.disc = discriminator.TimeDiscriminator(
                time_features=time_features, fc_activation=d_fc_activation,
                encoder_layers=d_encoder_layers, decoder_layers=d_decoder_layers, fc_layers=d_fc_layers,
                norm_epsilon=d_norm_epsilon, transformer_dropout_rate=d_transformer_dropout_rate,
                pre_out_dim=in_dim, out_dropout=d_out_dropout, recycle_fc_activ=d_recycle_fc_activ)
        else:  # mode_ == 'comb'
            self.disc = discriminator.Discriminator(
                embed_dim=embed_dim, n_heads=n_heads, kernel_size=d_kernel_size,
                fc_activation=d_fc_activation, encoder_layers=d_encoder_layers, decoder_layers=d_decoder_layers,
                fc_layers=d_fc_layers, norm_epsilon=d_norm_epsilon, transformer_dropout_rate=d_transformer_dropout_rate,
                pre_out_dim=in_dim, out_dropout=d_out_dropout, recycle_fc_activ=d_recycle_fc_activ)

    def callback_setting(self, model, optimizer, name, checkpoint_path, max_to_keep):
        self.ckpts[name] = tf.train.Checkpoint(model=model, optimizer=optimizer)
        self.ckpt_managers[name] = tf.train.CheckpointManager(self.ckpts[name], checkpoint_path, max_to_keep=max_to_keep)
        if self.ckpt_managers[name].latest_checkpoint:
            self.ckpts[name].restore(self.ckpt_managers[name].latest_checkpoint)
            print('Restored the latest {}'.format(name))

    def load_model(self, model_paths, max_to_keep=5):
        # generators
        if self.mode_ != 'time':
            self.embedder = gensim.models.Word2Vec.load(
                os.path.join(model_paths['chords_embedder'], 'chords_embedder.model'))
            self.callback_setting(
                self.chords_style, self.optimizer_gen, 'chords_style', model_paths['chords_style'], max_to_keep)
            self.callback_setting(
                self.chords_syn, self.optimizer_gen, 'chords_syn', model_paths['chords_syn'], max_to_keep)
        if self.mode_ != 'chords':
            self.callback_setting(
                self.time_style, self.optimizer_gen, 'time_style', model_paths['time_style'], max_to_keep)
            self.callback_setting(
                self.time_syn, self.optimizer_gen, 'time_syn', model_paths['time_syn'], max_to_keep)
        # discriminator
        disc_nm = '{}_disc'.format(self.mode_)
        self.callback_setting(
            self.disc, self.optimizer_disc, disc_nm, model_paths[disc_nm], max_to_keep)

    def set_trainable(self, train_chords_style=True, train_chords_syn=True,
                      train_time_style=True, train_time_syn=True, train_disc=True):
        if self.mode_ != 'time':
            self.train_chords_style = train_chords_style
            self.train_chords_syn = train_chords_syn
            util.model_trainable(self.chords_style, trainable=train_chords_style)
            util.model_trainable(self.chords_syn, trainable=train_chords_syn)
        if self.mode_ != 'chords':
            self.train_time_style = train_time_style
            self.train_time_syn = train_time_syn
            util.model_trainable(self.time_style, trainable=train_time_style)
            util.model_trainable(self.time_syn, trainable=train_time_syn)

        self.train_disc = train_disc
        util.model_trainable(self.disc, trainable=train_disc)

    def save_models(self):
        if self.mode_ != 'time':
            if self.train_chords_style:
                self.ckpt_managers['chords_style'].save()
            if self.train_chords_syn:
                self.ckpt_managers['chords_syn'].save()
        if self.mode_ != 'chords':
            if self.train_time_style:
                self.ckpt_managers['time_style'].save()
            if self.train_time_syn:
                self.ckpt_managers['time_syn'].save()
        if self.train_disc:
            self.ckpt_managers['{}_disc'.format(self.mode_)].save()

    def load_true_samples(self, step=30, batch_size=10, out_seq_len=64,
                          vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
                          pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''],
                          remove_same_chords=True):
        self.vel_norm = vel_norm
        self.tmps_norm = tmps_norm
        self.dur_norm = dur_norm
        self.batch_size = batch_size
        self.true_data = util.load_true_data_gan(
            out_seq_len, step=step, batch_size=batch_size, pths=pths, name_substr_list=name_substr_list,
            remove_same_chords=remove_same_chords)

    def process_chs_tr(self, chs_tr):
        # chs_tr: (batch, seq_len)
        embeddings = [[self.embedder.wv[ch_i.numpy().decode('UTF-8')].tolist() for ch_i in ch_ary] for ch_ary in chs_tr]
        return tf.constant(embeddings)  # (batch, seq_len, embed_dim)

    def process_tms_tr(self, tms_tr):
        # tms_tr: (batch, seq_len, time_features)
        return tf.strings.to_number(tms_tr)

    def gen_fake(self, ch_ltnt, tm_ltnt):
        # ch_ltnt: (batch, in_dim) np.array or None
        # tm_ltnt: (batch, in_dim) np.array or None
        # ch_pred: (batch, out_seq_len, embed_dim)
        ch_pred = self.chords_syn((self.chords_style(ch_ltnt), self.out_seq_len)) \
            if self.mode_ != 'time' else None
        # tm_pred: (batch, out_seq_len, time_features), softplus convert pred to > 0
        tm_pred = tf.keras.activations.softplus(self.time_syn((self.time_style(tm_ltnt), self.out_seq_len))) \
            if self.mode_ != 'chords' else None
        return ch_pred, tm_pred

    def gradient_penalty(self, interpolates):
        with tf.GradientTape() as tape_g:
            tape_g.watch(interpolates)
            _, pred_ch = self.disc(interpolates)  # (batch, 1)
            gradients_disc = tape_g.gradient(pred_ch, [interpolates])
        # gradients_disc[0]: (batch, out_seq_len, embed_dim or time_features)
        slopes = tf.sqrt(tf.math.reduce_sum(tf.math.square(gradients_disc[0])))
        gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        return gradient_penalty

    def train_step(self, inputs, to_recycle=True):
        # ch_ltnt: (batch, in_dim)
        # tm_ltnt: (batch, in_dim)
        # ch_ltnt2: (batch, in_dim)
        # tm_ltnt2: (batch, in_dim)
        # chs_tr: true sample chords (batch, out_seq_len, embed_dim) np.array or None
        # tms_tr: true sample time (batch, out_seq_len, time_features) np.array or None
        ch_ltnt, tm_ltnt, ch_ltnt2, tm_ltnt2, chs_tr, tms_tr = inputs

        # Step 1. train discriminator --------------------
        with tf.GradientTape() as tape:
            util.model_trainable(self.disc, trainable=True)  # unfreeze discriminator
            # chs_fk: (batch, out_seq_len, embed_dim)
            # tms_fk: (batch, out_seq_len, time_features)

            """
            ch_ltnt=tf.ones((50, 512))
            tm_ltnt =tf.ones((50, 512))
            embed_dim=16
            chs_tr = tf.ones((50, out_seq_len, embed_dim))
            batch_size=50
            gan_model

            """
            alpha = tf.random.uniform((self.batch_size, 1, 1), minval=0.0, maxval=1.0)
            chs_fk, tms_fk = self.gen_fake(ch_ltnt, tm_ltnt)

            if self.mode_ == 'chords':
                _, d_fk = self.disc(chs_fk)  # (batch, 1)
                tape.watch(d_fk)
                pre_out, d_tr = self.disc(chs_tr)  # (batch, 1)
                tape.watch(d_tr)
                interpolates = chs_tr + alpha * (chs_fk - chs_tr)  # (batch, out_seq_len, embed_dim)
                gp = self.gradient_penalty(interpolates)
            elif self.mode_ == 'time':
                _, d_fk = self.disc(tms_fk)
                tape.watch(d_fk)
                pre_out, d_tr = self.disc(tms_tr)
                tape.watch(d_tr)
                interpolates = tms_tr + alpha * (tms_fk - tms_tr)  # (batch, out_seq_len, time_features)
                gp = self.gradient_penalty(interpolates)
            else:  # self.mode_ == 'comb'
                _, d_fk = self.disc((chs_fk, tms_fk))
                tape.watch(d_fk)
                pre_out, d_tr = self.disc((chs_tr, tms_tr))
                tape.watch(d_tr)
                gp_ch = self.gradient_penalty(chs_tr + alpha * (chs_fk - chs_tr))
                gp_tm = self.gradient_penalty(tms_tr + alpha * (tms_fk - tms_tr))
                gp = (gp_ch + gp_tm) / tf.constant(2, dtype=tf.float32)  # todo: use a better weight for ch and tm!!!

            wloss_disc = tf.math.reduce_mean(d_fk, axis=-1) - tf.math.reduce_mean(d_tr, axis=-1) +\
                         self.LAMBDA * gp  # (batch)
            gradients_disc = tape.gradient(wloss_disc, self.disc.trainable_variables)
            """
            with tf.GradientTape() as tape:
                chs_fk, tms_fk = gan_model.gen_fake(ch_ltnt, tm_ltnt)
                _, d_fk = gan_model.disc(chs_fk)  # (batch, 1)
                tape.watch(d_fk)
                pre_out, d_tr = gan_model.disc(chs_tr)  # (batch, 1)
                tape.watch(d_tr)
                gp = gradient_penalty(chs_tr, chs_fk)
                wloss_disc = tf.math.reduce_mean(d_fk, axis=-1) - tf.math.reduce_mean(d_tr, axis=-1) + LAMBDA * gp  # must be (16, 16)
                gradients_disc = tape.gradient(wloss_disc, gan_model.disc.trainable_variables)
            
            """
            self.optimizer_disc.apply_gradients(zip(gradients_disc, self.disc.trainable_variables))
            self.train_loss_disc(wloss_disc)

        # Step 2: freeze discriminator and use the fake sample with true label to train generator ---------------
        with tf.GradientTape() as tape:
            util.model_trainable(self.disc, trainable=False)  # freeze discriminator
            # chs_fk2: (batch * 2, out_seq_len, embed_dim)
            # tms_fk2: (batch * 2, out_seq_len, time_features)
            chs_fk2, tms_fk2 = self.gen_fake(ch_ltnt2, tm_ltnt2)
            if self.mode_ == 'chords':
                d_inputs = chs_fk2
            elif self.mode_ == 'time':
                d_inputs = tms_fk2
            else:  # self.mode_ == 'comb'
                d_inputs = chs_fk2, tms_fk2
            _, d_fk2 = self.disc(d_inputs)  # (batch, 1)
            wloss_gen = -tf.reduce_mean(d_fk2, axis=-1)  # (batch * 2)
            variables_gen = []
            if self.mode_ != 'time':
                if self.train_chords_style:
                    variables_gen += self.chords_style.trainable_variables
                if self.train_chords_syn:
                    variables_gen += self.chords_syn.trainable_variables
            if self.mode_ != 'chords':
                if self.train_time_style:
                    variables_gen += self.time_style.trainable_variables
                if self.train_time_syn:
                    variables_gen += self.time_syn.trainable_variables
            gradients_gen = tape.gradient(wloss_gen, variables_gen)
            self.optimizer_gen.apply_gradients(zip(gradients_gen, variables_gen))
            self.train_loss_gen(wloss_gen)

        return wloss_disc, wloss_gen, pre_out  # loss_disc_tr, loss_disc_fk, loss_gen, pre_out

    def train(self, epochs=10, save_model_step=1, save_sample_step=1,
              print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=5, disc_lr=0.0001, gen_lr=0.1,
              optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
              result_path=result_path, out_seq_len=64, save_nsamples=3, recycle_step=2, LAMBDA=10):

        assert save_nsamples <= self.batch_size
        self.out_seq_len = out_seq_len
        self.optimizer_gen = optmzr(gen_lr)
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')
        self.optimizer_disc = optmzr(disc_lr)
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')
        self.LAMBDA = LAMBDA

        pre_out = None
        # ---------------------- training --------------------------
        for epoch in range(epochs):
            self.train_loss_disc.reset_states()
            self.train_loss_gen.reset_states()

            start = time.time()
            for i, true_samples in enumerate(self.true_data):
                if true_samples.shape[0] < self.batch_size:
                    # the last batch generated may be smaller, so go to next round
                    # the unselected samples may be selected next time because data will be shuffled
                    continue
                # true samples ---------------------------
                # chs_tr: (batch, seq_len) string
                # tms_tr: (batch, seq_len, time_features) string
                chs_tr, tms_tr = true_samples[:, :, 0], true_samples[:, :, 1:]
                # chs_tr: (batch, seq_len, embed_dim) float
                chs_tr = self.process_chs_tr(chs_tr)
                # tms_tr: (batch, seq_len, time_features) float
                tms_tr = self.process_tms_tr(tms_tr)

                # random vectors ---------------------------
                if recycle_step is None:
                    to_recycle = False
                else:
                    to_recycle = True if (i+1) % recycle_step == 0 else False

                # nt_ltnt: (batch, in_dim)
                ch_ltnt = pre_out if to_recycle & (pre_out is not None) \
                    else tf.random.normal((self.batch_size, self.in_dim),
                                          mean=0, stddev=self.latent_std, dtype=tf.float32)
                # tm_ltnt: (batch, in_dim)
                tm_ltnt = pre_out if to_recycle & (pre_out is not None) \
                    else tf.random.normal((self.batch_size, self.in_dim),
                                          mean=0, stddev=self.latent_std, dtype=tf.float32)
                # nt_ltnt2: (batch, in_dim)
                ch_ltnt2 = tf.random.normal((self.batch_size * 2, self.in_dim),
                                            mean=0, stddev=self.latent_std, dtype=tf.float32)
                # tm_ltnt2: (batch, in_dim)
                tm_ltnt2 = tf.random.normal((self.batch_size * 2, self.in_dim),
                                            mean=0, stddev=self.latent_std, dtype=tf.float32)

                # loss_disc_tr, loss_disc_fk, loss_gen, pre_out = self.train_step(
                #     (ch_ltnt, tm_ltnt, ch_ltnt2, tm_ltnt2, chs_tr, tms_tr), to_recycle=to_recycle)
                wloss_disc, wloss_gen, pre_out = self.train_step(
                    (ch_ltnt, tm_ltnt, ch_ltnt2, tm_ltnt2, chs_tr, tms_tr), to_recycle=to_recycle)

                if print_batch:
                    if (i + 1) % print_batch_step == 0:
                        print('Epoch {} Batch {}: gen_loss={:.4f}; disc_loss={:.4f}'.format(
                            epoch + 1, i + 1, wloss_gen.numpy().mean(), wloss_disc.numpy().mean()))

                # temporary -------------
                if (i + 1) % 100 == 0:
                    self.save_models()
                    mids = self.gen_music(save_nsamples)
                    for sp, mid in enumerate(mids):
                        file_name = os.path.join(result_path, self.mode_, 'ep{}_{}_{}.mid'.format(epoch + 1, i + 1, sp))
                        mid.save(file_name)
                    print('Saved {} fake samples'.format(save_nsamples))
                # temporary -------------

            if print_epoch:
                if (epoch + 1) % print_epoch_step == 0:
                    print('Epoch {}: Loss_gen={:.4f}, Loss_disc={:.4f}, Time used={:.4f}'.format(
                        epoch+1, self.train_loss_gen.result(), self.train_loss_disc.result(), time.time()-start))

            if (epoch+1) % save_model_step == 0:
                self.save_models()

            if (epoch+1) % save_sample_step == 0:
                mids = self.gen_music(save_nsamples)
                for sp, mid in enumerate(mids):
                    file_name = os.path.join(result_path, self.mode_, 'ep{}_{}_{}.mid'.format(epoch+1, i + 1, sp))
                    mid.save(file_name)
                print('Saved {} fake samples'.format(save_nsamples))

    def gen_music(self, nsamples):
        # nt_ltnt: (nsamples, in_dim)
        ch_ltnt = tf.random.normal((nsamples, self.in_dim), mean=0, stddev=self.latent_std, dtype=tf.float32)
        # tm_ltnt: (nsamples, in_dim)
        tm_ltnt = tf.random.normal((nsamples, self.in_dim), mean=0, stddev=self.latent_std, dtype=tf.float32)
        # ch_pred: (batch, out_seq_len, embed_dim) or None
        # tm_pred: (batch, out_seq_len, time_features) or None
        ch_pred, tm_pred = self.gen_fake(ch_ltnt, tm_ltnt)
        if tm_pred is None:
            tm_pred = np.array([[[1] * 3] * self.out_seq_len] * nsamples)  # (nsamples, out_seq_len, 3)
        if ch_pred is None:
            ch_pred = np.array([[['64']] * self.out_seq_len] * nsamples)  # (nsamples, out_seq_len, 1)
        else:
            ch_pred = np.array([[self.embedder.wv.similar_by_vector(ch, topn=1)[0][0] for ch in ch_p]
                                for ch_p in ch_pred.numpy()])[:, :, np.newaxis]  # str (nsamples, out_seq_len, 1)

        tm_pred = np.multiply(tm_pred, np.array([self.vel_norm, self.tmps_norm, self.dur_norm]))  # denormalise
        # ary: (nsamples, out_seq_len, 4)
        ary = np.concatenate([ch_pred.astype(object), tm_pred.astype(object)], axis=-1)
        mids = []
        for ary_i in ary:
            ary_i[:, 1] = np.clip(ary_i[:, 1], 0, 127)
            mid_i = preprocess.Conversion.arry2mid(ary_i)
            mids.append(mid_i)
        return mids



# Todo: use minibatch discrimination (https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b)
# Todo: use tanh at the last layer of generator
# Todo: schedule training like: if disc_loss > gen_loss ==> train gen, else train disc

"""
latent_std=1.0
in_dim=512
embed_dim=16
chstl_fc_layers=4
chstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1)
strt_dim=5
n_heads=4
init_knl=3
chsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1)
chsyn_encoder_layers=3
chsyn_decoder_layers=3
chsyn_fc_layers=3
chsyn_norm_epsilon=1e-6
chsyn_noise_std=0.5
chsyn_transformer_dropout_rate=0.2
time_features=3
tmstl_fc_layers=4
tmstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1)
tmsyn_encoder_layers=3
tmsyn_decoder_layers=3
tmsyn_fc_layers=3
tmsyn_norm_epsilon=1e-6
tmsyn_transformer_dropout_rate=0.2
tmsyn_noise_std=0.5
tmsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1)
d_kernel_size=3
d_encoder_layers=1
d_decoder_layers=1
d_fc_layers=3
d_norm_epsilon=1e-6
d_transformer_dropout_rate=0.2
d_fc_activation=tf.keras.activations.tanh
d_out_dropout=0.3
d_recycle_fc_activ=tf.keras.activations.elu
mode_='comb'
"""

