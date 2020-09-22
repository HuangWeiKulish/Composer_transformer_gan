import os
import time
import numpy as np
import tensorflow as tf
import util
import discriminator
import generator
import preprocess

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

path_base = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/'
model_paths = {k: os.path.join(path_base, k) for k in
               ['chords_style', 'chords_syn', 'time_style', 'time_syn',
                'chords_disc', 'time_disc', 'comb_disc']}
result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'

"""
tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_indexcer/chords_dict_mod.pkl'
tk = pkl.load(open(tk_path, 'rb'))
import json
len(json.loads(tk.get_config()['index_word']))  # 15000
"""


class GAN(tf.keras.models.Model):

    def __init__(self, in_dim=512, embed_dim=16,
                 chstl_fc_layers=4, chstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1),
                 strt_dim=3, chords_pool_size=15001, n_heads=4, init_knl=3, max_pos=None,
                 chsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                 chsyn_encoder_layers=3, chsyn_decoder_layers=3, chsyn_fc_layers=3, chsyn_norm_epsilon=1e-6,
                 chsyn_embedding_dropout_rate=0.2, chsyn_transformer_dropout_rate=0.2,
                 time_features=3, tmstl_fc_layers=4, tmstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1),
                 tmsyn_encoder_layers=3, tmsyn_decoder_layers=3, tmsyn_fc_layers=3, tmsyn_norm_epsilon=1e-6,
                 tmsyn_transformer_dropout_rate=0.2, tmsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                 d_kernel_size=3, d_encoder_layers=1, d_decoder_layers=1, d_fc_layers=3, d_norm_epsilon=1e-6,
                 d_transformer_dropout_rate=0.2, d_fc_activation=tf.keras.activations.tanh,
                 d_out_dropout=0.3, mode_='comb'):
        super(GAN, self).__init__()
        if mode_ in ['chords', 'comb']:
            assert embed_dim % n_heads == 0, 'make sure: embed_dim % chsyn_n_heads == 0'

        # ---------------------------------- settings ----------------------------------
        self.mode_ = mode_
        self.embed_dim = embed_dim
        self.time_features = time_features  # 3 for [velocity, velocity, time since last start, chords duration]
        self.in_dim = in_dim
        self.strt_dim = strt_dim
        # callback settings
        self.ckpts = dict()
        self.ckpt_managers = dict()
        # optimisers
        self.optimizer_gen = tf.keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.optimizer_disc = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # losses
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')

        # generators
        if self.mode_ != 'time':
            self.chords_style = generator.Mapping(fc_layers=chstl_fc_layers, activ=chstl_activ)
            self.chords_syn = generator.ChordsSynthesis(
                out_chords_pool_size=chords_pool_size, embed_dim=embed_dim, init_knl=init_knl, strt_dim=strt_dim,
                n_heads=n_heads, max_pos=max_pos, fc_activation=chsyn_fc_activation, encoder_layers=chsyn_encoder_layers,
                decoder_layers=chsyn_decoder_layers, fc_layers=chsyn_fc_layers, norm_epsilon=chsyn_norm_epsilon,
                embedding_dropout_rate=chsyn_embedding_dropout_rate,
                transformer_dropout_rate=chsyn_transformer_dropout_rate)
        if self.mode_ != 'chords':
            self.time_style = generator.Mapping(fc_layers=tmstl_fc_layers, activ=tmstl_activ)
            self.time_syn = generator.TimeSynthesis(
                time_features=time_features, init_knl=init_knl, strt_dim=strt_dim, fc_activation=tmsyn_fc_activation,
                encoder_layers=tmsyn_encoder_layers, decoder_layers=tmsyn_decoder_layers, fc_layers=tmsyn_fc_layers,
                norm_epsilon=tmsyn_norm_epsilon, transformer_dropout_rate=tmsyn_transformer_dropout_rate)

        # discriminator
        if self.mode_ == 'chords':
            self.disc = discriminator.ChordsDiscriminator(
                embed_dim=embed_dim, n_heads=n_heads, fc_activation=d_fc_activation,
                encoder_layers=d_encoder_layers, decoder_layers=d_decoder_layers, fc_layers=d_fc_layers,
                norm_epsilon=d_norm_epsilon, transformer_dropout_rate=d_transformer_dropout_rate,
                pre_out_dim=in_dim, out_dropout=d_out_dropout)
        elif self.mode_ == 'time':
            self.disc = discriminator.TimeDiscriminator(
                time_features=time_features, fc_activation=d_fc_activation,
                encoder_layers=d_encoder_layers, decoder_layers=d_decoder_layers, fc_layers=d_fc_layers,
                norm_epsilon=d_norm_epsilon, transformer_dropout_rate=d_transformer_dropout_rate,
                pre_out_dim=in_dim, out_dropout=d_out_dropout)
        else:  # self.mode_ == 'comb'
            self.disc = discriminator.Discriminator(
                embed_dim=embed_dim, n_heads=n_heads, kernel_size=d_kernel_size,
                fc_activation=d_fc_activation, encoder_layers=d_encoder_layers, decoder_layers=d_decoder_layers,
                fc_layers=d_fc_layers, norm_epsilon=d_norm_epsilon, transformer_dropout_rate=d_transformer_dropout_rate,
                pre_out_dim=in_dim, out_dropout=d_out_dropout)

    def load_true_samples(self, tk, step=30, batch_size=10, out_seq_len=64,
                          vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
                          pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['']):
        self.vel_norm = vel_norm
        self.tmps_norm = tmps_norm
        self.dur_norm = dur_norm
        self.batch_size = batch_size
        self.tk = tk
        self.true_data = util.load_true_data_gan(
            tk, out_seq_len, step=step, batch_size=batch_size, vel_norm=vel_norm,
            tmps_norm=tmps_norm, dur_norm=dur_norm, pths=pths, name_substr_list=name_substr_list)

    def callback_setting(self, model, optimizer, name, checkpoint_path, max_to_keep):
        self.ckpts[name] = tf.train.Checkpoint(model=model, optimizer=optimizer)
        self.ckpt_managers[name] = tf.train.CheckpointManager(self.ckpts[name], checkpoint_path, max_to_keep=max_to_keep)
        if self.ckpt_managers[name].latest_checkpoint:
            self.ckpts[name].restore(self.ckpt_managers[name].latest_checkpoint)
            print('Restored the latest {}'.format(name))

    def load_model(self, model_paths, max_to_keep=5):
        # generators
        if self.mode_ != 'time':
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

    def gen_fake(self, ch_ltnt, tm_ltnt, return_str=False, tk=None):
        # ch_ltnt: (batch, in_dim) np.array or None
        # tm_ltnt: (batch, in_dim) np.array or None
        # ch_pred: (batch, out_seq_len, embed_dim) if return_str is False else (batch, out_seq_len)
        ch_pred = self.chords_syn((self.chords_style(ch_ltnt), self.out_seq_len), return_str=return_str, tk=tk) \
            if self.mode_ != 'time' else None
        tm_pred = tf.keras.activations.softplus(self.time_syn((self.time_style(tm_ltnt), self.out_seq_len))) \
            if self.mode_ != 'chords' else None  # (batch, out_seq_len, time_features), softplus convert pred to > 0
        return ch_pred, tm_pred

    def train_discriminator(self, ch_ltnt, tm_ltnt, chs_tr, tms_tr, fake_mode=True, to_recycle=True):
        # ch_ltnt: (batch, in_dim) np.array or None
        # tm_ltnt: (batch, in_dim) np.array or None
        # chs_tr: true sample chords (batch, out_seq_len) np.array or None
        # tms_tr: true sample time (batch, out_seq_len, time_features) np.array or None

        # unfreeze discriminator
        util.model_trainable(self.disc, trainable=True)
        if fake_mode:
            chs_fk, tms_fk = self.gen_fake(ch_ltnt, tm_ltnt)
            if self.mode_ == 'chords':
                d_inputs = chs_fk
            elif self.mode_ == 'time':
                d_inputs = tms_fk
            else:  # self.mode_ == 'comb'
                d_inputs = chs_fk, tms_fk
            pred = self.disc(d_inputs, return_vec=False)  # (batch, 1)
            pre_out = None  # no recycle based on fake samples
        else:  # fake_mode is False
            if self.mode_ == 'chords':
                d_inputs = self.chords_syn.chords_emb(chs_tr)  # (batch, out_seq_len, embed_dim)
            elif self.mode_ == 'time':
                d_inputs = tms_tr
            else:  # self.mode_ == 'comb'
                chs_tr = self.chords_syn.chords_emb(chs_tr)  # (batch, out_seq_len, embed_dim)
                d_inputs = chs_tr, tms_tr
            if to_recycle:
                # pre_out: (batch, in_dim, 1)
                # pred: (batch, 1)
                pre_out, pred = self.disc(d_inputs, return_vec=True)
            else:  # recycle is False
                pred = self.disc(d_inputs, return_vec=False)
                pre_out = None

        lbl = tf.random.uniform(
            (self.batch_size, 1), minval=self.fake_label_smooth[0], maxval=self.fake_label_smooth[1],
            dtype=tf.dtypes.float32) if fake_mode \
            else tf.random.uniform(
            (self.batch_size, 1), minval=self.true_label_smooth[0], maxval=self.true_label_smooth[1],
            dtype=tf.dtypes.float32)

        loss_disc = self.d_loss_func(lbl, pred, from_logits=False, label_smoothing=0)  # todo: change???
        # if pre_out is not None:
        #     plt.hist(pre_out.numpy().flatten(), alpha=0.3, bins=100)
        #     plt.savefig('pre_out.png')

        return pre_out, loss_disc, self.disc.trainable_variables

    def train_generator(self, ch_ltnt, tm_ltnt):
        # ch_ltnt: (batch, in_dim) np.array or None
        # tm_ltnt: (batch, in_dim) np.array or None

        # freeze discriminator
        util.model_trainable(self.disc, trainable=False)

        chs_fk, tms_fk = self.gen_fake(ch_ltnt, tm_ltnt)
        if self.mode_ == 'chords':
            d_inputs = chs_fk
        elif self.mode_ == 'time':
            d_inputs = tms_fk
        else:  # self.mode_ == 'comb'
            d_inputs = chs_fk, tms_fk
        pred = self.disc(d_inputs, return_vec=False)  # (batch, 1)

        vbs = []
        if self.mode_ != 'time':
            if self.train_chords_style:
                vbs += self.chords_style.trainable_variables
            if self.train_chords_syn:
                vbs += self.chords_syn.trainable_variables
        if self.mode_ != 'chords':
            if self.train_time_style:
                vbs += self.time_style.trainable_variables
            if self.train_time_syn:
                vbs += self.time_syn.trainable_variables

        # label flipping with no label smoothing
        lbls = tf.ones((self.batch_size, 1), dtype=tf.float32)  # (batch, 1)
        loss_gen = self.g_loss_func(lbls, pred, from_logits=False, label_smoothing=0)
        return loss_gen, vbs

    def train_step(self, inputs, to_recycle=True):
        # ch_ltnt: (batch, in_dim, 1)
        # tm_ltnt: (batch, in_dim, 1)
        # ch_ltnt2: (batch, in_dim, 1)
        # tm_ltnt2: (batch, in_dim, 1)
        # chs_tr: true sample chords (batch, in_seq_len)
        # tms_tr: true sample time (batch, in_seq_len, time_features)
        ch_ltnt, tm_ltnt, ch_ltnt2, tm_ltnt2, chs_tr, tms_tr = inputs

        # Step 1. train discriminator on true samples --------------------
        with tf.GradientTape() as tape:
            pre_out, loss_disc_tr, variables_disc = self.train_discriminator(
                ch_ltnt=None, tm_ltnt=None, chs_tr=chs_tr, tms_tr=tms_tr, fake_mode=False, to_recycle=to_recycle)
            gradients_disc = tape.gradient(loss_disc_tr, variables_disc)
            self.optimizer_disc.apply_gradients(zip(gradients_disc, variables_disc))
            self.train_loss_disc(loss_disc_tr)

        # Step 2. train discriminator on fake samples --------------------
        with tf.GradientTape() as tape:
            _, loss_disc_fk, variables_disc = self.train_discriminator(
                ch_ltnt=ch_ltnt, tm_ltnt=tm_ltnt, chs_tr=None, tms_tr=None, fake_mode=True, to_recycle=to_recycle)
            gradients_disc_fk = tape.gradient(loss_disc_fk, variables_disc)
            self.optimizer_disc.apply_gradients(zip(gradients_disc_fk, variables_disc))
            self.train_loss_disc(loss_disc_fk)

        # Step 3: freeze discriminator and use the fake sample with true label to train generator ---------------
        with tf.GradientTape() as tape:
            loss_gen, variables_gen = self.train_generator(ch_ltnt2, tm_ltnt2)
            gradients_gen = tape.gradient(loss_gen, variables_gen)
            self.optimizer_gen.apply_gradients(zip(gradients_gen, variables_gen))
            self.train_loss_gen(loss_gen)

        return loss_disc_tr, loss_disc_fk, loss_gen, pre_out

    def train(self, tk, epochs=10, save_model_step=1, save_sample_step=1,
              print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=5, disc_lr=0.0001, gen_lr=0.1,
              optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
              g_loss_func=tf.keras.losses.binary_crossentropy, d_loss_func=tf.keras.losses.binary_crossentropy,
              result_path=result_path, out_seq_len=64, save_nsamples=3,
              vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
              true_label_smooth=(0.9, 1.0), fake_label_smooth=(0.0, 0.1), recycle_step=2):

        assert save_nsamples <= self.batch_size
        self.out_seq_len = out_seq_len
        self.optimizer_gen = optmzr(gen_lr)
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')
        self.optimizer_disc = optmzr(disc_lr)
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')
        self.true_label_smooth = true_label_smooth
        self.fake_label_smooth = fake_label_smooth
        self.g_loss_func = g_loss_func  # todo: change???
        self.d_loss_func = d_loss_func

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
                # chs_tr: (batch, seq_len)
                # tms_tr: (batch, seq_len, 3)
                chs_tr, tms_tr = true_samples[:, :, 0], true_samples[:, :, 1:]

                # random vectors ---------------------------
                if recycle_step is None:
                    to_recycle = False
                else:
                    to_recycle = True if (i+1) % recycle_step == 0 else False

                # nt_ltnt: (batch, in_dim, 1)
                ch_ltnt = pre_out if to_recycle & (pre_out is not None) \
                    else tf.random.normal((self.batch_size, self.in_dim), mean=0, stddev=1.0, dtype=tf.float32)
                # tm_ltnt: (batch, in_dim, 1)
                tm_ltnt = pre_out if to_recycle & (pre_out is not None) \
                    else tf.random.normal((self.batch_size, self.in_dim), mean=0, stddev=1.0, dtype=tf.float32)
                # nt_ltnt2: (batch, in_dim, 1)
                ch_ltnt2 = tf.random.normal((self.batch_size, self.in_dim), mean=0, stddev=1.0, dtype=tf.float32)
                # tm_ltnt2: (batch, in_dim, 1)
                tm_ltnt2 = tf.random.normal((self.batch_size, self.in_dim), mean=0, stddev=1.0, dtype=tf.float32)

                loss_disc_tr, loss_disc_fk, loss_gen, pre_out = self.train_step(
                    (ch_ltnt, tm_ltnt, ch_ltnt2, tm_ltnt2, chs_tr, tms_tr), to_recycle=to_recycle)

                if print_batch:
                    if (i + 1) % print_batch_step == 0:
                        print('Epoch {} Batch {}: gen_loss={:.4f}; disc_fake_loss={:.4f}, '
                              'disc_true_loss={:.4f};'.format(
                            epoch+1, i+1, loss_gen.numpy().mean(), loss_disc_fk.numpy().mean(),
                            loss_disc_tr.numpy().mean()))

                if (i + 1) % 100 == 0:
                    self.save_models()
                    mids = self.gen_music(save_nsamples, tk, vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm)
                    for sp, mid in enumerate(mids):
                        file_name = os.path.join(result_path, self.mode_, 'ep{}_{}_{}.mid'.format(epoch+1, i+1, sp))
                        mid.save(file_name)
                    print('Saved {} fake samples'.format(save_nsamples))

            if print_epoch:
                if (epoch + 1) % print_epoch_step == 0:
                    print('Epoch {}: Loss_gen={:.4f}, Loss_disc={:.4f}, Time used={:.4f}'.format(
                        epoch+1, self.train_loss_gen.result(), self.train_loss_disc.result(), time.time()-start))

            if (epoch+1) % save_model_step == 0:
                self.save_models()

            if (epoch+1) % save_sample_step == 0:
                mids = self.gen_music(save_nsamples, tk, vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm)
                for sp, mid in enumerate(mids):
                    file_name = os.path.join(result_path, self.mode_, 'ep{}_{}_{}.mid'.format(epoch+1, i + 1, sp))
                    mid.save(file_name)
                print('Saved {} fake samples'.format(save_nsamples))

    def gen_music(self, save_nsamples, tk, vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm):
        # nt_ltnt: (save_nsamples, in_dim)
        ch_ltnt = tf.random.normal((save_nsamples, self.in_dim), mean=0, stddev=1.0, dtype=tf.float32)
        # tm_ltnt: (save_nsamples, in_dim)
        tm_ltnt = tf.random.normal((save_nsamples, self.in_dim), mean=0, stddev=1.0, dtype=tf.float32)
        # ch_pred: (batch, out_seq_len, embed_dim) or None
        # tm_pred: (batch, out_seq_len, time_features) or None
        ch_pred, tm_pred = self.gen_fake(ch_ltnt, tm_ltnt, return_str=True, tk=tk)
        if self.mode_ != 'time':
            tm_pred = np.array([[[1] * 3] * self.out_seq_len] * save_nsamples)
        if self.mode_ != 'chords':
            ch_pred = np.array([[['64']] * self.out_seq_len] * save_nsamples)

        tm_pred = np.multiply(tm_pred, np.array([vel_norm, tmps_norm, dur_norm]))  # denormalise
        # ary: (save_nsamples, out_seq_len, 4)
        ary = np.concatenate([ch_pred[:, :, np.newaxis].astype(object), tm_pred.astype(object)], axis=-1)
        mids = []
        for ary_i in ary:
            ary_i[:, 1] = np.clip(ary_i[:, 1], 0, 127)
            mid_i = preprocess.Conversion.arry2mid(ary_i)
            mids.append(mid_i)
        return mids



# Todo: use minibatch discrimination (https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b)
# Todo: use tanh at the last layer of generator
# Todo: schedule training like: if disc_loss > gen_loss ==> train gen, else train disc
# Todo: use different loss function max(logD)

"""
in_dim=512
embed_dim=16
chstl_fc_layers=4
chstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1)
strt_dim=3
chords_pool_size=15000
n_heads=4
init_knl=3
max_pos=None
chsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1)
chsyn_encoder_layers=3
chsyn_decoder_layers=3
chsyn_fc_layers=3
chsyn_norm_epsilon=1e-6
chsyn_embedding_dropout_rate=0.2
chsyn_transformer_dropout_rate=0.2
time_features=3
tmstl_fc_layers=4
tmstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1)
tmsyn_encoder_layers=3
tmsyn_decoder_layers=3
tmsyn_fc_layers=3
tmsyn_norm_epsilon=1e-6
tmsyn_transformer_dropout_rate=0.2
tmsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1)
d_kernel_size=3
d_encoder_layers=1
d_decoder_layers=1
d_fc_layers=3
d_norm_epsilon=1e-6
d_transformer_dropout_rate=0.2
d_fc_activation=tf.keras.activations.tanh
d_out_dropout=0.3
mode_='comb'
"""

