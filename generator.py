import numpy as np
import tensorflow as tf
import transformer

tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class AdaInstanceNormalization(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-3):
        super(AdaInstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs, training=None):
        input_shape = tf.keras.backend.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))
        beta = inputs[1]
        gamma = inputs[2]
        mean = tf.keras.backend.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = tf.keras.backend.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev
        return normed * gamma + beta


class Mapping(tf.keras.layers.Layer):

    def __init__(self, fc_layers=4, dropout_rate=0.2, epsilon=0.0001, activ=tf.keras.layers.LeakyReLU(alpha=0.1)):
        super(Mapping, self).__init__()
        self.ln = tf.keras.layers.LayerNormalization(epsilon=epsilon, axis=0)
        self.fcs = tf.keras.models.Sequential(
            [tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation=activ),
                                         tf.keras.layers.Dropout(dropout_rate)])
             for _ in range(fc_layers)])

    def call(self, x, training=None):
        # x: (batch, in_dim)
        out = self.ln(x)  # (batch, in_dim)
        out = self.fcs(out)  # (batch, in_dim)
        return out


class ChordsEmbedder(tf.keras.layers.Layer):

    def __init__(self, chords_pool_size, max_pos, embed_dim=16, dropout_rate=0.2):
        super(ChordsEmbedder, self).__init__()
        self.chords_pool_size = chords_pool_size
        self.max_pos = max_pos
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.embedder = tf.keras.layers.Embedding(chords_pool_size, embed_dim)

    def call(self, x_in, training=None):
        # x_in: (batch, time_in): chords index
        chords_emb = self.embedder(x_in)  # (batch, time_in, embed_dim)
        if self.max_pos is not None:
            pos_emb = ChordsEmbedder.positional_encoding(self.max_pos, self.embed_dim)  # (1, max_pos, embed_dim)
            chords_emb += pos_emb[:, : x_in.get_shape()[1], :]  # (batch, time_in, embed_dim)
        if self.dropout_rate is not None:
            chords_emb = tf.keras.layers.Dropout(self.dropout_rate)(chords_emb)
        return chords_emb  # (batch, time_in, embed_dim)

    @staticmethod
    def get_angles(pos, i, embed_dim):
        # reference: https://www.tensorflow.org/tutorials/text/transformer
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(max_pos, embed_dim):
        # reference: https://www.tensorflow.org/tutorials/text/transformer
        angle_rads = ChordsEmbedder.get_angles(
            np.arange(max_pos)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)


class ChordsSynthesis(tf.keras.models.Model):

    def __init__(self, strt_token_id=15001, out_chords_pool_size=15001, embed_dim=16, init_knl=3, strt_dim=5,
                 n_heads=4, max_pos=800, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3,
                 norm_epsilon=1e-6, embedding_dropout_rate=0.2, transformer_dropout_rate=0.2):
        super(ChordsSynthesis, self).__init__()
        self.strt_token_id = strt_token_id   # strt_token_id = tk.word_index['<start>']
        self.strt_dim = strt_dim
        self.embed_dim = embed_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.init_fc = tf.keras.layers.Dense(embed_dim)
        self.init_ext = tf.keras.layers.Conv1DTranspose(filters=embed_dim, kernel_size=init_knl, strides=strt_dim)
        self.chords_emb = ChordsEmbedder(
            chords_pool_size=out_chords_pool_size, max_pos=max_pos, embed_dim=embed_dim,
            dropout_rate=embedding_dropout_rate)
        self.chords_extend = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_chords_pool_size=out_chords_pool_size,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)
        self.b_fc = tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')
        self.g_fc = tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='ones')
        self.noise_en_fc = tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')
        self.noise_de1_fc = tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')
        self.noise_de2_fc = tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')

    def call(self, inputs, training=None, mask=None, return_str=False, tk=None):
        # styl: (batch, in_dim)
        styl, out_seq_len = inputs

        x_en = self.init_fc(styl)  # (batch, embed_dim)
        x_en = self.init_ext(x_en[:, tf.newaxis, :])  # (batch, strt_dim, embed_dim)
        self.x_de = self.chords_emb(tf.constant(
            [[self.strt_token_id]] * styl.shape[0], dtype=tf.float32))  # (batch, 1, embed_dim)
        beta = self.b_fc(styl)[:, tf.newaxis, :]  # (batch, 1, embed_dim)
        gamma = self.g_fc(styl)[:, tf.newaxis, :]  # (batch, 1, embed_dim)
        # (batch, out_seq_len, embed_dim)
        x_out = self.extend_x(x_en, out_seq_len, beta, gamma, return_str=return_str, tk=tk)
        return x_out

    def extend_x(self, x_en, out_seq_len, beta, gamma, return_str=False, tk=None):
        # x_en: (batch, strt_dim, embed_dim)
        result = []  # (out_seq_len, batch)
        for i in range(out_seq_len):
            # noise_en: list of noise (batch, en_time_in, embed_dim), list length = encoder_layers
            # noise_de_1: list of noise (batch, out_seq_len, embed_dim), list length = decoder_layers
            # noise_de_2: list of noise (batch, out_seq_len, embed_dim), list length = decoder_layers
            noise_en = self.noise_en_fc(tf.random.uniform(
                (self.encoder_layers, x_en.shape[0], x_en.shape[1], self.embed_dim)))
            noise_de_1 = self.noise_de1_fc(tf.random.uniform(
                (self.decoder_layers, x_en.shape[0], 1, self.embed_dim)))
            noise_de_2 = self.noise_de2_fc(tf.random.uniform(
                (self.decoder_layers, x_en.shape[0], 1, self.embed_dim)))
            # x_out: (batch, 1, out_chords_pool_size)
            x_out, _ = self.chords_extend((x_en, self.x_de, None, None),
                                          noise_en=noise_en, noise_de_1=noise_de_1, noise_de_2=noise_de_2)
            # translate prediction to text
            pred = tf.argmax(x_out, -1)  # (batch, 1)
            if return_str:
                result.append([tk.index_word[pd.numpy()[-1] + 1] for pd in pred])
            pred_ = self.chords_emb(pred[:, -1][:, tf.newaxis])
            pred_ = AdaInstanceNormalization()([pred_, beta, gamma])
            x_en = tf.concat((x_en, pred_), axis=1)  # (batch, en_time_in+1, embed_dim)
        if return_str:
            return np.transpose(result, (1, 0)).astype(object)  # (batch, out_seq_len), string format
        return x_en[:, self.strt_dim:, :]  # (batch, out_seq_len, embed_dim)


class TimeSynthesis(tf.keras.models.Model):

    def __init__(self, time_features=3, init_knl=3, strt_dim=5, fc_activation="relu", encoder_layers=1,
                 decoder_layers=1, fc_layers=3, norm_epsilon=1e-6, transformer_dropout_rate=0.2):
        super(TimeSynthesis, self).__init__()
        self.time_features = time_features
        self.strt_dim = strt_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.embed_dim = time_features

        self.init_fc = tf.keras.layers.Dense(time_features)
        self.init_ext = tf.keras.layers.Conv1DTranspose(filters=time_features, kernel_size=init_knl, strides=strt_dim)
        self.chords_extend = transformer.Transformer(
            embed_dim=time_features, n_heads=1, out_chords_pool_size=time_features,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)
        self.b_fc = tf.keras.layers.Dense(
            time_features, kernel_initializer='he_normal', bias_initializer='zeros')
        self.g_fc = tf.keras.layers.Dense(
            time_features, kernel_initializer='he_normal', bias_initializer='ones')
        self.noise_en_fc = tf.keras.layers.Dense(
            time_features, kernel_initializer='he_normal', bias_initializer='zeros')
        self.noise_de1_fc = tf.keras.layers.Dense(
            time_features, kernel_initializer='he_normal', bias_initializer='zeros')
        self.noise_de2_fc = tf.keras.layers.Dense(
            time_features, kernel_initializer='he_normal', bias_initializer='zeros')

    def call(self, inputs, training=None, mask=None):
        # styl: (batch, in_dim)
        styl, out_seq_len = inputs
        x_en = self.init_fc(styl)  # (batch, time_features)
        x_en = self.init_ext(x_en[:, tf.newaxis, :])  # (batch, strt_dim, time_features)
        self.x_de = tf.ones((styl.shape[0], 1, self.time_features))  # (batch, 1, time_features)
        beta = self.b_fc(styl)[:, tf.newaxis, :]  # (batch, 1, time_features)
        gamma = self.g_fc(styl)[:, tf.newaxis, :]  # (batch, 1, time_features)
        x_out = self.extend_x(x_en, out_seq_len, beta, gamma)  # (batch, out_seq_len, time_features)
        return x_out

    def extend_x(self, x_en, out_seq_len, beta, gamma):
        # x_en: (batch, strt_dim, time_features)
        # beta: (batch, 1, time_features)
        # gamma: (batch, 1, time_features)
        for i in range(out_seq_len):
            # noise_en: list of noise (batch, en_time_in, time_features), list length = encoder_layers
            # noise_de_1: list of noise (batch, out_seq_len, time_features), list length = decoder_layers
            # noise_de_2: list of noise (batch, out_seq_len, time_features), list length = decoder_layers
            noise_en = self.noise_en_fc(tf.random.uniform(
                (self.encoder_layers, x_en.shape[0], x_en.shape[1], self.embed_dim)))
            noise_de_1 = self.noise_de1_fc(tf.random.uniform(
                (self.decoder_layers, x_en.shape[0], 1, self.embed_dim)))
            noise_de_2 = self.noise_de2_fc(tf.random.uniform(
                (self.decoder_layers, x_en.shape[0], 1, self.embed_dim)))
            # x_out: (batch, 1, time_features)
            x_out, _ = self.chords_extend((x_en, self.x_de, None, None),
                                          noise_en=noise_en, noise_de_1=noise_de_1, noise_de_2=noise_de_2)

            pred = AdaInstanceNormalization()([x_out, beta, gamma])
            x_en = tf.concat((x_en, pred), axis=1)  # (batch, en_time_in+1, time_features)
        return x_en[:, self.strt_dim:, :]  # (batch, out_seq_len, time_features)

