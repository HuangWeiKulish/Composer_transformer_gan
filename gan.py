import os
import time
import numpy as np
import pickle as pkl
import tensorflow as tf
import util
import discriminator
import generator
import preprocess
import transformer

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3
out_seq_len_list = (8, 16, 32, 64)

# some layers
ini_layer_path = {
    'chords': '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/ini_chords',
    'time': '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/ini_time'}
const_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/constant'
chords_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_embedder'
style_paths = {
    'chords': '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_style',
    'time': '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_style'}

# synthesis
chords_syn_path_base = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_syn'
chords_syn_model_name_list = ['ch_b_fcs', 'ch_g_fcs', 'ch_n_cv1s', 'ch_up_trs']
chords_syn_paths = {k: {k_: os.path.join(chords_syn_path_base, k, str(k_))
                        for k_ in out_seq_len_list}
                    for k in chords_syn_model_name_list}
time_syn_path_base = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_syn'
time_syn_model_name_list = ['tm_b_fcs', 'tm_g_fcs', 'tm_n_cv1s', 'tm_up_trs']
time_syn_paths = {k: {k_: os.path.join(time_syn_path_base, k, str(k_))
                      for k_ in out_seq_len_list}
                  for k in time_syn_model_name_list}
# discriminator
disc_paths = {
    'chords': '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_discriminator',
    'time': '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_discriminator',
    'both': '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/discriminator'}
# random sample result
result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'

"""
tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_indexcer/notes_dict_final.pkl'
tk = pkl.load(open(tk_path, 'rb'))
"""


def syn_init_layer(consttile, styl, ini_layer):
    # how in ['chords', 'time']
    # consttile: (batch, strt_dim, in_dim)
    # styl: (batch, in_dim, 1)
    # ini_layer: tf.keras.layers.Dense
    ini = tf.matmul(consttile, styl)  # (batch, strt_dim, 1)
    return ini_layer(ini)  # (batch, strt_dim, embed_dim) for chords or (batch, strt_dim, time_features) for time


def extend_chords(x_en, out_seq_len, strt_token_id, chords_emb, up_tr, tk, return_str=False):
    # x_en: (batch_size, in_seq_len, embed_dim)
    # chords_emb: generator.ChordsEmbedder
    # up_tr: transformer.Transformer
    batch_size = x_en.shape[0]
    x_de = tf.constant([[strt_token_id]] * batch_size, dtype=tf.float32)
    x_de = chords_emb(x_de)  # (batch_size, 1, embed_dim)
    result = []
    for i in range(out_seq_len):
        mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
        mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
        # x_out: (batch_size, 1, out_chords_pool_size)
        x_out, _ = up_tr((x_en, x_de, mask_padding, mask_lookahead))
        # translate prediction to text
        pred = tf.argmax(x_out, -1)  # (batch_size, 1)
        x_de = tf.concat((x_de, chords_emb(pred[:, -1][:, tf.newaxis])), axis=1)
        if return_str:
            pred = [nid for nid in pred.numpy()[:, -1]]  # len = batch, take the last prediction
            result.append([tk.index_word[pd + 1] for pd in pred])  # return chords string
    if return_str:
        return np.transpose(np.array(result, dtype=object), (1, 0))  # (batch, out_seq_len)
    return x_de[:, 1:, :]  # (batch_size, out_seq_len, embed_dim)


def extend_time(x_en, out_seq_len, time_features, up_tr):
    # x_en: (batch_size, in_seq_len, time_features)
    # up_tr: transformer.Transformer
    batch_size = x_en.shape[0]
    x_de = tf.zeros((batch_size, 1, time_features), dtype=tf.float32)  # (batch_size, 1, time_features)
    for i in range(out_seq_len):
        mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
        mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
        # x_out: (batch_size, 1, out_chords_pool_size)
        x_out, _ = up_tr((x_en, x_de, mask_padding, mask_lookahead))
        x_de = tf.concat((x_de, x_out[:, -1][:, tf.newaxis]), axis=1)
    return x_de[:, 1:, :]  # (batch_size, out_seq_len, time_features)


def chords_synthesis_bloc(conv_in, style_in, noise, out_seq_len, strt_token_id,
                          chords_emb, b_fc, g_fc, up_tr, n_cv1, activ, tk, return_str):
    # conv_in: (batch, updated strt_dim, embed_dim)
    # style_in: (batch, style_dim, 1)
    # noise: (batch, embed_dim, 1)
    # chords_emb: generator.ChordsEmbedder(chords_pool_size=out_chords_pool_size, max_pos=max_pos, embed_dim=embed_dim,
    #                                      dropout_rate=embedding_dropout_rate)
    # b_fc: tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')
    # g_fc: tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='ones')
    # up_tr: transformer.Transformer(
    #             embed_dim=embed_dim, n_heads=n_heads, out_chords_pool_size=out_chords_pool_size,
    #             encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
    #             norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation,
    #             out_positive=False)
    b = b_fc(tf.transpose(style_in, perm=(0, 2, 1)))  # (batch, 1, embed_dim)
    g = g_fc(tf.transpose(style_in, perm=(0, 2, 1)))  # (batch, 1, embed_dim)
    n = tf.transpose(n_cv1(noise), perm=(0, 2, 1))  # (batch, out_seq_len, embed_dim)
    out = extend_chords(conv_in, out_seq_len, strt_token_id, chords_emb, up_tr,
                        tk, return_str)  # (batch, out_seq_len, embed_dim)
    if not return_str:
        out = tf.add(out, n)  # (batch, out_seq_len, embed_dim)
        out = generator.AdaInstanceNormalization()([out, b, g])  # (batch, out_seq_len, embed_dim)
        out = activ(out)  # (batch, out_seq_len, embed_dim)

    # b2 = b_fc1(tf.transpose(style_in, perm=(0, 2, 1)))  # (batch, 1, embed_dim)
    # g2 = g_fc1(tf.transpose(style_in, perm=(0, 2, 1)))  # (batch, 1, embed_dim)
    # n2 = tf.transpose(n_cv1_1(noise), perm=(0, 2, 1))  # (batch, out_seq_len, embed_dim)
    # out = extend_chords(out, out_seq_len, strt_token_id, chords_emb, up_tr1)  # (batch, out_seq_len, embed_dim)
    # out = tf.add(out, n2)  # (batch, out_seq_len, embed_dim)
    # out = generator.AdaInstanceNormalization()([out, b2, g2])  # (batch, out_seq_len, embed_dim)
    # out = activ(out)  # (batch, out_seq_len, embed_dim)
    return out


def time_synthesis_bloc(conv_in, style_in, noise, out_seq_len, time_features,
                        b_fc, g_fc, up_tr, n_cv1s, activ):
    # conv_in: (batch, updated strt_dim, time_features)
    # style_in: (batch, style_dim, 1)
    # noise: (time_features, 1)

    # b_fc: tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')
    # g_fc: tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='ones')
    # up_tr: transformer.Transformer(
    #             embed_dim=embed_dim, n_heads=n_heads, out_chords_pool_size=out_chords_pool_size,
    #             encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
    #             norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation,
    #             out_positive=False)
    b = b_fc(tf.transpose(style_in, perm=(0, 2, 1)))  # (batch, 1, time_features)
    g = g_fc(tf.transpose(style_in, perm=(0, 2, 1)))  # (batch, 1, time_features)
    n = tf.transpose(n_cv1s(noise), perm=(0, 2, 1))  # (batch, out_seq_len, time_features)
    out = extend_time(conv_in, out_seq_len, time_features, up_tr)  # (batch, out_seq_len, time_features)
    out = tf.add(out, n)  # (batch, out_seq_len, time_features)
    out = generator.AdaInstanceNormalization()([out, b, g])  # (batch, out_seq_len, time_features)
    out = activ(out)  # (batch, out_seq_len, time_features)

    # b2 = b_fc1(tf.transpose(style_in, perm=(0, 2, 1)))  # (batch, 1, time_features)
    # g2 = g_fc1(tf.transpose(style_in, perm=(0, 2, 1)))  # (batch, 1, time_features)
    # n2 = tf.transpose(n_cv1_1(noise), perm=(0, 2, 1))  # (batch, out_seq_len, time_features)
    # out = extend_time(out, out_seq_len, time_features, up_tr)  # (batch, out_seq_len, time_features)
    # out = tf.add(out, n2)  # (batch, out_seq_len, time_features)
    # out = generator.AdaInstanceNormalization()([out, b2, g2])  # (batch, out_seq_len, time_features)
    # out = activ(out)  # (batch, out_seq_len, time_features)
    return out


def synthsis_ch(nt_conv_in, nt_styl, batch_size, embed_dim, strt_token_id, out_seq_len_list,
                chords_emb, ch_b_fcs, ch_g_fcs, ch_up_trs, ch_n_cv1s, activ, tk, return_str=False):
    # nt_conv_in: (batch, strt_dim, embed_dim)
    # nt_styl: (batch, style_dim, 1)
    nt_conv_in_ = nt_conv_in
    l = len(out_seq_len_list) - 1
    i = 0
    return_str_ = False
    for sql, b_fc, g_fc, up_tr, n_cv1 in zip(out_seq_len_list, ch_b_fcs, ch_g_fcs, ch_up_trs, ch_n_cv1s):
        # noise: (batch, embed_dim, 1)
        noise = tf.random.normal((batch_size, embed_dim, 1), mean=0, stddev=1.0, dtype=tf.float32)
        if return_str & (i == l):
            return_str_ = True
        nt_conv_in_ = chords_synthesis_bloc(nt_conv_in_, nt_styl, noise, sql, strt_token_id,
                                            chords_emb, b_fc, g_fc, up_tr, n_cv1, activ, tk, return_str_)
        i += 1
    return nt_conv_in_  # (batch, out_seq_len_list[-1], embed_dim)


def synthsis_tm(tm_conv_in, tm_styl, batch_size, time_features, out_seq_len_list,
                tm_b_fcs, tm_g_fcs, tm_up_trs, tm_n_cv1s, activ):
    # tm_conv_in: (batch, strt_dim, time_features)
    # tm_styl: (batch, style_dim, 1)
    tm_conv_in_ = tm_conv_in
    for sql, b_fc, g_fc, up_tr, n_cv1 in zip(out_seq_len_list, tm_b_fcs, tm_g_fcs, tm_up_trs, tm_n_cv1s):
        # noise: (batch, time_features, 1)
        noise = tf.random.normal((batch_size, time_features, 1), mean=0, stddev=1.0, dtype=tf.float32)
        tm_conv_in_ = time_synthesis_bloc(tm_conv_in_, tm_styl, noise, sql, time_features,
                                          b_fc, g_fc, up_tr, n_cv1, activ)
    tm_conv_in_ = tf.keras.activations.softplus(tm_conv_in_)  # convert values to postive
    return tm_conv_in_  # (batch, out_seq_len_list[-1], time_features)


def generate_chords(batch_size, embed_dim, strt_token_id, consttile_ch, out_seq_len_list,
                    style_dim, chords_style, chords_ini, chords_emb, ch_b_fcs, ch_g_fcs, ch_up_trs, ch_n_cv1s,
                    activ, tk, out_midi=True):
    ch_ltnt = tf.random.normal((batch_size, style_dim, 1))  # (batch, style_dim, 1)
    nt_styl = chords_style(ch_ltnt)  # (batch, style_dim, 1)
    nt_conv_in = syn_init_layer(consttile_ch, nt_styl, chords_ini)  # (batch, strt_dim, embed_dim)
    chs = synthsis_ch(nt_conv_in, nt_styl, batch_size, embed_dim, strt_token_id, out_seq_len_list,
                      chords_emb, ch_b_fcs, ch_g_fcs, ch_up_trs, ch_n_cv1s,
                      activ, tk, return_str=True)  # (batch, out_seq_len_list[-1])  contains string
    if out_midi:
        # (batch, out_seq_len_list[-1], time_features)
        tms = np.array([[[vel_norm, tmps_norm, dur_norm]] * out_seq_len_list[-1]] * batch_size)
        # ary: (out_seq_len, out_seq_len_list[-1], 4)
        ary = np.concatenate([chs[:, :, np.newaxis].astype(object), tms.astype(object)], axis=-1)
        mids = []
        for ary_i in ary:
            ary_i = ary_i[(ary_i[:, 0] != '<start>') & (ary_i[:, 0] != '<end>')]
            mid_i = preprocess.Conversion.arry2mid(ary_i)
            mids.append(mid_i)
        return mids
    return chs


def generate_time(batch_size, time_features, consttile_tm, out_seq_len_list,
                  style_dim, time_style, time_ini, tm_b_fcs, tm_g_fcs, tm_up_trs, tm_n_cv1s,
                  activ, out_midi=True):
    tm_ltnt = tf.random.normal((batch_size, style_dim, 1))  # (batch, style_dim, 1)
    tm_styl = time_style(tm_ltnt)  # (batch, style_dim, 1)
    tm_conv_in = syn_init_layer(consttile_tm, tm_styl, time_ini)  # (batch, strt_dim, time_features)
    tms = synthsis_tm(tm_conv_in, tm_styl, batch_size, time_features, out_seq_len_list,
                      tm_b_fcs, tm_g_fcs, tm_up_trs, tm_n_cv1s, activ)  # (batch, out_seq_len_list[-1], time_features)
    tms = tms.numpy() * np.array([vel_norm, tmps_norm, dur_norm])  # un-normalise
    tms[:, 0] = np.clip(tms[:, 0], 0, 127)  # squeeze velocity within limit
    if out_midi:
        chs = np.array([[['64']] * out_seq_len_list[-1]] * batch_size)
        # ary: (out_seq_len, out_seq_len_list[-1], 4)
        ary = np.concatenate([chs.astype(object), tms.astype(object)], axis=-1)
        mids = []
        for ary_i in ary:
            mid_i = preprocess.Conversion.arry2mid(ary_i)
            mids.append(mid_i)
        return mids
    return tms


def generate_music(mode_, batch_size, embed_dim, style_dim, time_features, strt_token_id, consttile_ch, consttile_tm,
                   out_seq_len_list, tk,
                   chords_style, chords_ini, chords_emb, ch_b_fcs, ch_g_fcs, ch_up_trs, ch_n_cv1s, ch_activ,
                   time_style, time_ini, tm_b_fcs, tm_g_fcs, tm_up_trs, tm_n_cv1s, tm_activ,
                   out_midi=True):
    if mode_ == 'chords':
        return generate_chords(
            batch_size, embed_dim, strt_token_id, consttile_ch, out_seq_len_list,
            style_dim, chords_style, chords_ini, chords_emb, ch_b_fcs, ch_g_fcs, ch_up_trs, ch_n_cv1s,
            ch_activ, tk, out_midi=out_midi)
    if mode_ == 'time':
        return generate_time(
            batch_size, time_features, consttile_tm, out_seq_len_list,
            style_dim, time_style, time_ini, tm_b_fcs, tm_g_fcs, tm_up_trs, tm_n_cv1s,
            tm_activ, out_midi=out_midi)
    # mode_ == 'both'
    chs = generate_chords(
            batch_size, embed_dim, strt_token_id, consttile_ch, out_seq_len_list,
            style_dim, chords_style, chords_ini, chords_emb, ch_b_fcs, ch_g_fcs, ch_up_trs, ch_n_cv1s,
            ch_activ, tk, out_midi=False)  # (batch, out_seq_len_list[-1])
    tms = generate_time(
        batch_size, time_features, consttile_tm, out_seq_len_list,
        style_dim, time_style, time_ini, tm_b_fcs, tm_g_fcs, tm_up_trs, tm_n_cv1s,
        tm_activ, out_midi=False)  # (batch, out_seq_len_list[-1], time_features)

    ary = np.concatenate([chs[:, :, np.newaxis].astype(object), tms.astype(object)], axis=-1)
    ary = ary[(ary[:, 0] != '<start>') & (ary[:, 0] != '<end>')]
    if out_midi:
        mids = []
        for ary_i in ary:
            mid_i = preprocess.Conversion.arry2mid(ary_i)
            mids.append(mid_i)
        return mids
    return ary


class GAN(tf.keras.models.Model):

    def __init__(self, out_seq_len_list=out_seq_len_list, embed_dim=16, time_features=3, in_dim=512, n_heads=4,
                 strt_dim=4, strt_token_id=15001, chords_pool_size=15002, chords_max_pos=800, mode_='both',
                 chstl_fc_layers=4, chstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1),
                 tmstl_fc_layers=4, tmstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1), embedding_dropout_rate=0.2,
                 chsyn_kernel_size=3, chsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                 chsyn_encoder_layers=3, chsyn_decoder_layers=3, chsyn_fc_layers=3,
                 chsyn_norm_epsilon=1e-6, chsyn_transformer_dropout_rate=0.2,
                 chsyn_activ=tf.keras.layers.LeakyReLU(alpha=0.1),
                 tmsyn_kernel_size=3, tmsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                 tmsyn_encoder_layers=3, tmsyn_decoder_layers=3,
                 tmsyn_fc_layers=3, tmsyn_norm_epsilon=1e-6, tmsyn_transformer_dropout_rate=0.2,
                 tmsyn_activ=tf.keras.layers.LeakyReLU(alpha=0.1),
                 d_kernel_size=3, d_encoder_layers=1, d_decoder_layers=1, d_fc_layers=3, d_norm_epsilon=1e-6,
                 d_transformer_dropout_rate=0.2, d_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                 d_out_dropout=0.3):
        super(GAN, self).__init__()
        if mode_ in ['chords', 'both']:
            assert embed_dim % n_heads == 0, 'make sure: embed_dim % chsyn_n_heads == 0'

        # ---------------------------------- settings ----------------------------------
        self.mode_ = mode_
        self.out_seq_len_list = out_seq_len_list
        self.embed_dim = embed_dim
        self.time_features = time_features  # 3 for [velocity, velocity, time since last start, chords duration]
        self.in_dim = in_dim
        self.strt_dim = strt_dim
        self.strt_token_id = strt_token_id  # strt_token_id = tk.word_index['<start>']
        self.n_heads = n_heads
        self.chords_pool_size = chords_pool_size

        # chords synthesis network
        self.chsyn_kernel_size, self.chsyn_fc_activation, self.chsyn_encoder_layers, self.chsyn_decoder_layers, \
            self.chsyn_fc_layers, self.chsyn_norm_epsilon, self.chsyn_transformer_dropout_rate, self.chsyn_activ = \
            chsyn_kernel_size, chsyn_fc_activation, chsyn_encoder_layers, chsyn_decoder_layers, \
            chsyn_fc_layers, chsyn_norm_epsilon, chsyn_transformer_dropout_rate, chsyn_activ

        # time synthesis network
        self.tmsyn_kernel_size, self.tmsyn_fc_activation, self.tmsyn_encoder_layers, self.tmsyn_decoder_layers, \
            self.tmsyn_fc_layers, self.tmsyn_norm_epsilon, self.tmsyn_transformer_dropout_rate, self.tmsyn_activ = \
            tmsyn_kernel_size, tmsyn_fc_activation, tmsyn_encoder_layers, tmsyn_decoder_layers, \
            tmsyn_fc_layers, tmsyn_norm_epsilon, tmsyn_transformer_dropout_rate, tmsyn_activ

        # discriminator settings
        self.d_kernel_size, self.d_encoder_layers, self.d_decoder_layers, self.d_fc_layers, self.d_norm_epsilon, \
            self.d_transformer_dropout_rate, self.d_fc_activation, self.d_out_dropout = \
            d_kernel_size, d_encoder_layers, d_decoder_layers, d_fc_layers, d_norm_epsilon, \
            d_transformer_dropout_rate, d_fc_activation, d_out_dropout

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
        if self.mode_ != 'time':
            # constant layer init: const_ch: (1, strt_dim, in_dim)
            self.const_ch = tf.Variable(np.ones((1, strt_dim, in_dim)), dtype=tf.float32)
            # chords embedder
            self.chords_emb = generator.ChordsEmbedder(
                chords_pool_size=chords_pool_size, max_pos=chords_max_pos, embed_dim=embed_dim,
                dropout_rate=embedding_dropout_rate)
            # style generator
            self.chords_style = generator.Mapping(fc_layers=chstl_fc_layers, activ=chstl_activ)
            # initial layer of synthesis networks
            self.chords_ini = tf.keras.layers.Dense(embed_dim) if self.mode_ != 'time' else None
            # synthesis networks
            self.init_syn_chords()

        if self.mode_ != 'chords':
            # constant layer init: const_tm: (1, strt_dim, in_dim)
            self.const_tm = tf.Variable(np.ones((1, strt_dim, in_dim)), dtype=tf.float32)
            # style generator
            self.time_style = generator.Mapping(fc_layers=tmstl_fc_layers, activ=tmstl_activ)
            # initial layer of synthesis networks
            self.time_ini = tf.keras.layers.Dense(time_features)
            # synthesis networks
            self.init_syn_time()

        self.init_discriminator()

    def init_syn_chords(self):
        self.ch_b_fcs = [tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')
            for _ in self.out_seq_len_list]
        self.ch_g_fcs = [
            tf.keras.layers.Dense(self.embed_dim, kernel_initializer='he_normal', bias_initializer='ones')
            for _ in self.out_seq_len_list]
        self.ch_n_cv1s = [tf.keras.layers.Conv1D(
            sql, kernel_size=self.chsyn_kernel_size, padding='same', kernel_initializer='zeros',
            bias_initializer='zeros')
            for sql in self.out_seq_len_list]
        self.ch_up_trs = [transformer.Transformer(
            embed_dim=self.embed_dim, n_heads=self.n_heads, out_chords_pool_size=self.chords_pool_size,
            encoder_layers=self.chsyn_encoder_layers, decoder_layers=self.chsyn_decoder_layers,
            fc_layers=self.chsyn_fc_layers, norm_epsilon=self.chsyn_norm_epsilon,
            dropout_rate=self.chsyn_transformer_dropout_rate, fc_activation=self.chsyn_fc_activation)
            for _ in self.out_seq_len_list]

    def init_syn_time(self):
        self.tm_b_fcs = [tf.keras.layers.Dense(
            self.time_features, kernel_initializer='he_normal', bias_initializer='zeros')
            for _ in self.out_seq_len_list]
        self.tm_g_fcs = [tf.keras.layers.Dense(
            self.time_features, kernel_initializer='he_normal', bias_initializer='ones')
            for _ in self.out_seq_len_list]
        self.tm_n_cv1s = [tf.keras.layers.Conv1D(
            sql, kernel_size=self.tmsyn_kernel_size, padding='same', kernel_initializer='zeros',
            bias_initializer='zeros')
            for sql in self.out_seq_len_list]
        self.tm_up_trs = [transformer.Transformer(
            embed_dim=self.time_features, n_heads=1, out_chords_pool_size=self.time_features,
            encoder_layers=self.tmsyn_encoder_layers, decoder_layers=self.tmsyn_decoder_layers,
            fc_layers=self.tmsyn_fc_layers,
            norm_epsilon=self.tmsyn_norm_epsilon, dropout_rate=self.tmsyn_transformer_dropout_rate,
            fc_activation=self.tmsyn_fc_activation)
            for _ in self.out_seq_len_list]

    def init_discriminator(self):
        if self.mode_ == 'chords':
            self.disc = discriminator.ChordsDiscriminator(
                embed_dim=self.embed_dim, n_heads=self.n_heads, fc_activation=self.d_fc_activation,
                encoder_layers=self.d_encoder_layers, decoder_layers=self.d_decoder_layers, fc_layers=self.d_fc_layers,
                norm_epsilon=self.d_norm_epsilon, transformer_dropout_rate=self.d_transformer_dropout_rate,
                pre_out_dim=self.in_dim, out_dropout=self.d_out_dropout)
        elif self.mode_ == 'time':
            self.disc = discriminator.TimeDiscriminator(
                time_features=self.time_features, fc_activation=self.d_fc_activation,
                encoder_layers=self.d_encoder_layers, decoder_layers=self.d_decoder_layers, fc_layers=self.d_fc_layers,
                norm_epsilon=self.d_norm_epsilon, transformer_dropout_rate=self.d_transformer_dropout_rate,
                pre_out_dim=self.in_dim, out_dropout=self.d_out_dropout)
        else:  # self.mode_ == 'both'
            self.disc = discriminator.Discriminator(
                embed_dim=self.embed_dim, n_heads=self.n_heads, kernel_size=self.d_kernel_size,
                fc_activation=self.d_fc_activation, encoder_layers=self.d_encoder_layers,
                decoder_layers=self.d_decoder_layers, fc_layers=self.d_fc_layers, norm_epsilon=self.d_norm_epsilon,
                transformer_dropout_rate=self.d_transformer_dropout_rate,
                pre_out_dim=self.in_dim, out_dropout=self.d_out_dropout)

    def load_true_samples(self, tk, step=30, batch_size=10, vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
                          pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['']):
        self.vel_norm = vel_norm
        self.tmps_norm = tmps_norm
        self.dur_norm = dur_norm
        self.batch_size = batch_size
        self.tk = tk
        self.true_data = util.load_true_data_gan(
            tk, self.out_seq_len_list[-1], step=step, batch_size=batch_size, vel_norm=vel_norm,
            tmps_norm=tmps_norm, dur_norm=dur_norm, pths=pths, name_substr_list=name_substr_list)

    def callback_setting(self, model, optimizer, name, checkpoint_path, max_to_keep):
        self.ckpts[name] = tf.train.Checkpoint(model=model, optimizer=optimizer)
        self.ckpt_managers[name] = tf.train.CheckpointManager(self.ckpts[name], checkpoint_path, max_to_keep=max_to_keep)
        if self.ckpt_managers[name].latest_checkpoint:
            self.ckpts[name].restore(self.ckpt_managers[name].latest_checkpoint)
            print('Restored the latest {}'.format(name))

    def load_model(self, const_path=const_path, chords_emb_path=chords_emb_path, ini_layer_path=ini_layer_path,
                   style_paths=style_paths, chords_syn_paths=chords_syn_paths, time_syn_paths=time_syn_paths,
                   disc_paths=disc_paths,max_to_keep=5):
        # ---------------------- constant --------------------------
        self.const_path = const_path
        try:
            # const: (1, strt_dim, in_dim)
            const_ch = pkl.load(open(os.path.join(const_path, 'const_ch.pkl'), 'rb'))  # numpy array
            self.const_ch = tf.Variable(const_ch, dtype=tf.float32)  # convert to variable
            print('Restored the latest const_ch')
        except:
            pass
        try:
            # const: (1, strt_dim, in_dim)
            const_tm = pkl.load(open(os.path.join(const_path, 'const_tm.pkl'), 'rb'))  # numpy array
            self.const_tm = tf.Variable(const_tm, dtype=tf.float32)  # convert to variable
            print('Restored the latest const_tm')
        except:
            pass

        # ---------------------- call back setting --------------------------
        if self.mode_ != 'time':
            # chords embedder
            self.callback_setting(
                self.chords_emb, self.optimizer_gen, 'chords_emb', chords_emb_path, max_to_keep)
            # initial layers
            self.callback_setting(
                self.chords_ini, self.optimizer_gen, 'chords_ini', ini_layer_path['chords'], max_to_keep)
            # style generator
            self.callback_setting(
                self.chords_style, self.optimizer_gen, 'chords_style', style_paths['chords'], max_to_keep)
            # chords sythesis
            for i, sql in enumerate(self.out_seq_len_list):
                self.callback_setting(
                    self.ch_b_fcs[i], self.optimizer_gen,
                    'ch_b_fcs__{}'.format(sql), chords_syn_paths['ch_b_fcs'][sql], max_to_keep)
                self.callback_setting(
                    self.ch_g_fcs[i], self.optimizer_gen,
                    'ch_g_fcs__{}'.format(sql), chords_syn_paths['ch_g_fcs'][sql], max_to_keep)
                self.callback_setting(
                    self.ch_n_cv1s[i], self.optimizer_gen,
                    'ch_n_cv1s__{}'.format(sql), chords_syn_paths['ch_n_cv1s'][sql], max_to_keep)
                self.callback_setting(
                    self.ch_up_trs[i], self.optimizer_gen,
                    'ch_up_trs__{}'.format(sql), chords_syn_paths['ch_up_trs'][sql], max_to_keep)

        if self.mode_ != 'chords':
            # initial layers
            self.callback_setting(
                self.time_ini, self.optimizer_gen, 'time_ini', ini_layer_path['time'], max_to_keep)
            # style generator
            self.callback_setting(
                self.time_style, self.optimizer_gen, 'time_style', style_paths['time'], max_to_keep)
            # time sythesis
            for i, sql in enumerate(self.out_seq_len_list):
                self.callback_setting(
                    self.tm_b_fcs[i], self.optimizer_gen,
                    'tm_b_fcs__{}'.format(sql), time_syn_paths['tm_b_fcs'][sql], max_to_keep)
                self.callback_setting(
                    self.tm_g_fcs[i], self.optimizer_gen,
                    'tm_g_fcs__{}'.format(sql), time_syn_paths['tm_g_fcs'][sql], max_to_keep)
                self.callback_setting(
                    self.tm_n_cv1s[i], self.optimizer_gen,
                    'tm_n_cv1s__{}'.format(sql), time_syn_paths['tm_n_cv1s'][sql], max_to_keep)
                self.callback_setting(
                    self.tm_up_trs[i], self.optimizer_gen,
                    'tm_up_trs__{}'.format(sql), time_syn_paths['tm_up_trs'][sql], max_to_keep)

        # discriminator
        self.callback_setting(
            self.disc, self.optimizer_disc, 'disc__{}'.format(self.mode_), disc_paths[self.mode_], max_to_keep)

    def set_trainable(self, train_emb=True, train_chords_ini=True, train_time_ini=True,
                      train_style_ch=True, train_style_tm=True, train_syn_ch={k: True for k in out_seq_len_list},
                      train_syn_tm={k: True for k in out_seq_len_list}, train_disc=True):
        if self.mode_ != 'time':
            self.train_emb = train_emb
            util.model_trainable(self.chords_emb, trainable=train_emb)
            self.train_chords_ini = train_chords_ini
            self.train_style_ch = train_style_ch
            self.train_syn_ch = train_syn_ch
            util.model_trainable(self.chords_ini, trainable=train_chords_ini)
            util.model_trainable(self.chords_style, trainable=train_style_ch)
            for i, sql in enumerate(self.out_seq_len_list):
                util.model_trainable(self.ch_b_fcs[i], trainable=train_syn_ch[sql])
                util.model_trainable(self.ch_g_fcs[i], trainable=train_syn_ch[sql])
                util.model_trainable(self.ch_n_cv1s[i], trainable=train_syn_ch[sql])
                util.model_trainable(self.ch_up_trs[i], trainable=train_syn_ch[sql])

        if self.mode_ != 'chords':
            self.train_time_ini = train_time_ini
            self.train_style_tm = train_style_tm
            self.train_syn_tm = train_syn_tm
            util.model_trainable(self.time_ini, trainable=train_time_ini)
            util.model_trainable(self.time_style, trainable=train_style_tm)
            for i, sql in enumerate(self.out_seq_len_list):
                util.model_trainable(self.tm_b_fcs[i], trainable=train_syn_tm[sql])
                util.model_trainable(self.tm_g_fcs[i], trainable=train_syn_tm[sql])
                util.model_trainable(self.tm_n_cv1s[i], trainable=train_syn_tm[sql])
                util.model_trainable(self.tm_up_trs[i], trainable=train_syn_tm[sql])

        self.train_disc = train_disc
        util.model_trainable(self.disc, trainable=train_disc)

    def save_models(self):
        if self.mode_ != 'time':
            pkl.dump(self.const_ch.numpy(), open(os.path.join(self.const_path, 'const_ch.pkl'), 'wb'))
            self.ckpt_managers['chords_emb'].save()
            self.ckpt_managers['chords_ini'].save()
            self.ckpt_managers['chords_style'].save()
            for sql in self.out_seq_len_list:
                for nm in ['ch_b_fcs__{}'.format(sql), 'ch_g_fcs__{}'.format(sql), 'ch_n_cv1s__{}'.format(sql),
                           'ch_up_trs__{}'.format(sql)]:
                    self.ckpt_managers[nm].save()

        if self.mode_ != 'chords':
            pkl.dump(self.const_tm.numpy(), open(os.path.join(self.const_path, 'const_tm.pkl'), 'wb'))
            self.ckpt_managers['time_ini'].save()
            self.ckpt_managers['time_style'].save()
            for sql in self.out_seq_len_list:
                for nm in ['tm_b_fcs__{}'.format(sql), 'tm_g_fcs__{}'.format(sql), 'tm_n_cv1s__{}'.format(sql),
                           'tm_up_trs__{}'.format(sql)]:
                    self.ckpt_managers[nm].save()

        self.ckpt_managers['disc__{}'.format(self.mode_)].save()

    def train_discriminator(self, ch_ltnt, tm_ltnt, chs_tr, tms_tr, fake_mode=True):
        # ch_ltnt: (batch, in_dim, 1) np.array or None
        # tm_ltnt: (batch, in_dim, 1) np.array or None
        # chs_tr: true sample chords (batch, in_seq_len) np.array or None
        # tms_tr: true sample time (batch, in_seq_len, time_features) np.array or None
        
        # unfreeze discriminator
        util.model_trainable(self.disc, trainable=True)

        # (batch, 1, embed_dim) or (batch, 1, time_features)
        de_in = self.chords_emb(tf.constant([[self.strt_token_id]] * self.batch_size, dtype=tf.float32)) \
            if self.mode_ != 'time' \
            else tf.constant([[[0] * self.time_features]] * self.batch_size, dtype=tf.float32)

        if fake_mode:
            if self.mode_ != 'time':
                ch_styl = self.chords_style(ch_ltnt)  # (batch, style_dim, 1)
                ch_conv_in = syn_init_layer(self.consttile_ch, ch_styl, self.chords_ini)  # (batch, strt_dim, embed_dim)
                chs_fk = synthsis_ch(
                    ch_conv_in, ch_styl, self.batch_size, self.embed_dim, self.strt_token_id, self.out_seq_len_list,
                    self.chords_emb, self.ch_b_fcs, self.ch_g_fcs, self.ch_up_trs, self.ch_n_cv1s,
                    self.chsyn_activ, tk=None, return_str=False)  # (batch, out_seq_len_list[-1], embed_dim)
            if self.mode_ != 'chords':
                tm_styl = self.time_style(tm_ltnt)  # (batch, style_dim, 1)
                tm_conv_in = syn_init_layer(self.const_tm, tm_styl, self.time_ini)  # (batch, strt_dim, time_features)
                tms_fk = synthsis_tm(
                    tm_conv_in, tm_styl, self.batch_size, self.time_features, self.out_seq_len_list,
                    self.tm_b_fcs, self.tm_g_fcs, self.tm_up_trs, self.tm_n_cv1s,
                    self.tmsyn_activ)  # (batch, out_seq_len_list[-1], time_features)
            if self.mode_ == 'chords':
                d_inputs = chs_fk, de_in
            elif self.mode_ == 'time':
                d_inputs = tms_fk, de_in
            else:  # self.mode_ == 'both'
                d_inputs = chs_fk, tms_fk, de_in
            pred = self.disc(d_inputs, return_vec=False)  # (batch, 1)
            pre_out = None  # no recycle based on fake samples

        else:  # fake_mode is False
            if self.mode_ == 'chords':
                chs_tr = self.chords_emb(chs_tr)  # (batch, out_seq_len_list[-1], embed_dim)
                d_inputs = chs_tr, de_in
            elif self.mode_ == 'time':
                d_inputs = tms_tr, de_in
            else:  # self.mode_ == 'both'
                chs_tr = self.chords_emb(chs_tr)  # (batch, out_seq_len_list[-1], embed_dim)
                d_inputs = chs_tr, tms_tr, de_in
            if self.recycle:
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

        loss_disc = tf.keras.losses.binary_crossentropy(lbl, pred, from_logits=False, label_smoothing=0)
        return pre_out, loss_disc, self.disc.trainable_variables

    def train_generator(self, ch_ltnt, tm_ltnt):
        # ch_ltnt: (batch, in_seq_len, 16)
        # tm_ltnt: (batch, in_seq_len, 1)

        # freeze discriminator
        if self.train_disc:
            util.model_trainable(self.disc, trainable=False)

        de_in = self.chords_emb(tf.constant([[self.strt_token_id]] * self.batch_size, dtype=tf.float32)) \
            if self.mode_ != 'time' \
            else tf.constant([[[0] * self.time_features]] * self.batch_size, dtype=tf.float32)
        vbs = []
        if self.mode_ != 'time':
            ch_styl = self.chords_style(ch_ltnt)  # (batch, style_dim, 1)
            ch_conv_in = syn_init_layer(self.consttile_ch, ch_styl, self.chords_ini)  # (batch, strt_dim, embed_dim)
            chs_fk = synthsis_ch(
                ch_conv_in, ch_styl, self.batch_size, self.embed_dim, self.strt_token_id, self.out_seq_len_list,
                self.chords_emb, self.ch_b_fcs, self.ch_g_fcs, self.ch_up_trs, self.ch_n_cv1s,
                self.chsyn_activ, tk=None, return_str=False)  # (batch, out_seq_len_list[-1], embed_dim)
            vbs += self.chords_emb.trainable_variables + self.chsyn_activ.trainable_variables
            for i in range(len(self.out_seq_len_list)):
                vbs += self.ch_b_fcs[i].trainable_variables + self.ch_g_fcs[i].trainable_variables + \
                       self.ch_up_trs[i].trainable_variables + self.ch_n_cv1s[i].trainable_variables

        if self.mode_ != 'chords':
            tm_styl = self.time_style(tm_ltnt)  # (batch, style_dim, 1)
            tm_conv_in = syn_init_layer(self.const_tm, tm_styl, self.time_ini)  # (batch, strt_dim, time_features)
            tms_fk = synthsis_tm(
                tm_conv_in, tm_styl, self.batch_size, self.time_features, self.out_seq_len_list,
                self.tm_b_fcs, self.tm_g_fcs, self.tm_up_trs, self.tm_n_cv1s,
                self.tmsyn_activ)  # (batch, out_seq_len_list[-1], time_features)
            vbs += self.tmsyn_activ.trainable_variables
            for i in range(len(self.out_seq_len_list)):
                vbs += self.tm_b_fcs[i].trainable_variables + self.tm_g_fcs[i].trainable_variables + \
                       self.tm_up_trs[i].trainable_variables + self.tm_n_cv1s[i].trainable_variables

        if self.mode_ == 'chords':
            d_inputs = chs_fk, de_in
        elif self.mode_ == 'time':
            d_inputs = tms_fk, de_in
        else:  # self.mode_ == 'both'
            d_inputs = chs_fk, tms_fk, de_in
        pred = self.disc(d_inputs, return_vec=False)  # (batch, 1)

        # label flipping with no label smoothing
        lbls = tf.ones((self.batch_size, 1), dtype=tf.float32)  # (batch, 1)
        loss_gen = tf.keras.losses.binary_crossentropy(lbls, pred, from_logits=False, label_smoothing=0)
        return loss_gen, vbs

    def train_step(self, inputs):
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
                ch_ltnt=None, tm_ltnt=None, chs_tr=chs_tr, tms_tr=tms_tr, fake_mode=False)
            gradients_disc = tape.gradient(loss_disc_tr, variables_disc)
            self.optimizer_disc.apply_gradients(zip(gradients_disc, variables_disc))
            self.train_loss_disc(loss_disc_tr)

        # Step 2. train discriminator on fake samples --------------------
        with tf.GradientTape() as tape:
            _, loss_disc_fk, variables_disc = self.train_discriminator(
                ch_ltnt=ch_ltnt, tm_ltnt=tm_ltnt, chs_tr=None, tms_tr=None, fake_mode=True)
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
              result_path=result_path, save_nsamples=3,
              true_label_smooth=(0.9, 1.0), fake_label_smooth=(0.0, 0.1), recycle=True):

        self.optimizer_gen = optmzr(gen_lr)
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')
        self.optimizer_disc = optmzr(disc_lr)
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')
        self.true_label_smooth = true_label_smooth
        self.fake_label_smooth = fake_label_smooth
        self.recycle = recycle

        if self.mode_ != 'time':
            # consttile_ch: (batch, strt_dim, in_dim)
            self.consttile_ch = tf.tile(self.const_ch, tf.constant([self.batch_size, 1, 1], tf.int32))
        if self.mode_ != 'chords':
            # consttile_tm: (batch, strt_dim, in_dim)
            self.consttile_tm = tf.tile(self.const_tm, tf.constant([self.batch_size, 1, 1], tf.int32))

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
                # nt_ltnt: (batch, in_dim, 1)
                nt_ltnt = pre_out if recycle & (pre_out is not None) \
                    else tf.random.normal((self.batch_size, self.in_dim, 1), mean=0, stddev=1.0, dtype=tf.float32)
                # tm_ltnt: (batch, in_dim, 1)
                tm_ltnt = pre_out if recycle & (pre_out is not None) \
                    else tf.random.normal((self.batch_size, self.in_dim, 1), mean=0, stddev=1.0, dtype=tf.float32)
                # nt_ltnt2: (batch, in_dim, 1)
                nt_ltnt2 = tf.random.normal((self.batch_size, self.in_dim, 1), mean=0, stddev=1.0, dtype=tf.float32)
                # tm_ltnt2: (batch, in_dim, 1)
                tm_ltnt2 = tf.random.normal((self.batch_size, self.in_dim, 1), mean=0, stddev=1.0, dtype=tf.float32)

                loss_disc_tr, loss_disc_fk, loss_gen, pre_out = self.train_step(
                    (nt_ltnt, tm_ltnt, nt_ltnt2, tm_ltnt2, chs_tr, tms_tr))

                if print_batch:
                    if (i + 1) % print_batch_step == 0:
                        print('Epoch {} Batch {}: gen_loss={:.4f}; disc_fake_loss={:.4f}, '
                              'disc_true_loss={:.4f};'.format(
                            epoch+1, i+1, loss_gen.numpy().mean(), loss_disc_fk.numpy().mean(),
                            loss_disc_tr.numpy().mean()))

                if (i + 1) % 500 == 0:
                    self.save_models()
                    mids = self.gen_music(save_nsamples, tk)
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
                mids = self.gen_music(save_nsamples, tk)
                for sp, mid in enumerate(mids):
                    file_name = os.path.join(result_path, self.mode_, 'ep{}_{}_{}.mid'.format(epoch+1, i + 1, sp))
                    mid.save(file_name)
                print('Saved {} fake samples'.format(save_nsamples))

    def gen_music(self, save_nsamples, tk):
        if self.mode_ != 'time':
            consttile_ch_, chords_style_, chords_ini_, chords_emb_, \
            ch_b_fcs_, ch_g_fcs_, ch_up_trs_, ch_n_cv1s_, chsyn_activ_ = \
                self.consttile_ch, self.chords_style, self.chords_ini, self.chords_emb, \
                self.ch_b_fcs, self.ch_g_fcs, self.ch_up_trs, self.ch_n_cv1s, self.chsyn_activ
        else:
            consttile_ch_, chords_style_, chords_ini_, chords_emb_, \
            ch_b_fcs_, ch_g_fcs_, ch_up_trs_, ch_n_cv1s_, chsyn_activ_ = \
                None, None, None, None, None, None, None, None, None
        if self.mode_ != 'chords':
            consttile_tm_, time_style_, time_ini_, tm_b_fcs_, tm_g_fcs_, \
            tm_up_trs_, tm_n_cv1s_, tmsyn_activ_ = \
                self.consttile_tm, self.time_style, self.time_ini, self.tm_b_fcs, self.tm_g_fcs, \
                self.tm_up_trs, self.tm_n_cv1s, self.tmsyn_activ
        else:
            consttile_tm_, time_style_, time_ini_, tm_b_fcs_, tm_g_fcs_, \
            tm_up_trs_, tm_n_cv1s_, tmsyn_activ_ = None, None, None, None, None, None, None, None

        mids = generate_music(
            self.mode_, save_nsamples, self.embed_dim, self.in_dim, self.time_features,
            self.strt_token_id, consttile_ch_, consttile_tm_, self.out_seq_len_list, tk, chords_style_,
            chords_ini_, chords_emb_, ch_b_fcs_, ch_g_fcs_, ch_up_trs_, ch_n_cv1s_, chsyn_activ_,
            time_style_, time_ini_, tm_b_fcs_, tm_g_fcs_, tm_up_trs_, tm_n_cv1s_, tmsyn_activ_,
            out_midi=True)

        return mids


# Todo: use minibatch discrimination (https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b)
# Todo: use tanh at the last layer of generator
# Todo: schedule training like: if disc_loss > gen_loss ==> train gen, else train disc
# Todo: use different loss function max(logD)
# try recycle gan

"""
out_seq_len_list=(8, 16, 32, 64)
embed_dim=16
n_heads=4
in_dim=512
time_features=3
strt_dim=4
strt_token_id=15001
chords_pool_size=15002
chords_max_pos=800
chstl_fc_layers=4
chstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1)
tmstl_fc_layers=4
tmstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1)
chsyn_kernel_size=3
chsyn_n_heads=4
chsyn_fc_activation="relu"
chsyn_encoder_layers=3
chsyn_decoder_layers=3
chsyn_fc_layers=3
chsyn_norm_epsilon=1e-6
embedding_dropout_rate=0.2
chsyn_transformer_dropout_rate=0.2
chsyn_activ=tf.keras.layers.LeakyReLU(alpha=0.1)
tmsyn_kernel_size=3
tmsyn_fc_activation="relu"
tmsyn_encoder_layers=3
tmsyn_decoder_layers=3
tmsyn_fc_layers=3
tmsyn_norm_epsilon=1e-6
tmsyn_transformer_dropout_rate=0.2
tmsyn_activ=tf.keras.layers.LeakyReLU(alpha=0.1)
d_kernel_size=3
d_encoder_layers=1
d_decoder_layers=1
d_fc_layers=3
d_norm_epsilon=1e-6
d_transformer_dropout_rate=0.2
d_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1)
d_out_dropout=0.3


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

