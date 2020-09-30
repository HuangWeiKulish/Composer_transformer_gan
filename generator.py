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


class Mapping(tf.keras.models.Model):

    def __init__(self, fc_layers=4, dropout_rate=0.2, activ=tf.keras.layers.LeakyReLU(alpha=0.1)):
        super(Mapping, self).__init__()
        self.fcs = tf.keras.models.Sequential(
            [tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation=activ),
                                         #tf.keras.layers.GaussianNoise(0.3),
                                         #tf.keras.layers.Dropout(dropout_rate),
                                         tf.keras.layers.BatchNormalization(
                                             momentum=0.99, epsilon=0.001, center=True, scale=True,
                                             beta_initializer='zeros', gamma_initializer='ones',)])  # Todo: add normalisarion? need dropout?
             for _ in range(fc_layers)])

    def call(self, x, training=None, mask=None):
        # x: (batch, in_dim)
        return self.fcs(x)  # (batch, in_dim)


class ChordsSynthesis(tf.keras.models.Model):

    def __init__(self, embed_dim=16, init_knl=3, strt_dim=5, n_heads=4, fc_activation=tf.keras.activations.relu,
                 encoder_layers=2, decoder_layers=2, fc_layers=3, norm_epsilon=1e-6, transformer_dropout_rate=0.2,
                 noise_std=0.5):
        super(ChordsSynthesis, self).__init__()
        self.strt_dim = strt_dim
        self.embed_dim = embed_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.noise_std = noise_std
        self.init_fc = tf.keras.layers.Dense(embed_dim)
        self.init_ext = tf.keras.layers.Conv1DTranspose(filters=embed_dim, kernel_size=init_knl, strides=strt_dim)
        self.chords_extend = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            fc_layers=fc_layers, norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate,
            fc_activation=fc_activation)
        self.b_fc = tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')
        self.g_fc = tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='ones')
        self.noise_en_fc = tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')
        self.noise_de_fc = tf.keras.layers.Dense(embed_dim, kernel_initializer='he_normal', bias_initializer='zeros')

    def call(self, inputs, training=None, mask=None):
        # styl: (batch, in_dim)
        styl, out_seq_len = inputs
        x_en = self.init_fc(styl)  # (batch, embed_dim)
        # x_en = tf.keras.layers.GaussianNoise(0.5)(x_en)  # add noise
        x_en = self.init_ext(x_en[:, tf.newaxis, :])  # (batch, strt_dim, embed_dim)
        beta = self.b_fc(styl)[:, tf.newaxis, :]  # (batch, 1, embed_dim)
        gamma = self.g_fc(styl)[:, tf.newaxis, :]  # (batch, 1, embed_dim)
        # (batch, out_seq_len, embed_dim)
        x_out = self.extend_x(x_en, out_seq_len, beta, gamma)
        return x_out

    def extend_x(self, x_en, out_seq_len, beta, gamma):
        # x_en: (batch, strt_dim, embed_dim)
        x_de = tf.ones((x_en.shape[0], 1, self.embed_dim))  # (batch, 1, embed_dim)
        for i in range(out_seq_len):
            # noise_en: list of noise (batch, en_time_in, embed_dim), list length = encoder_layers
            # noise_de: list of noise (batch, out_seq_len, embed_dim), list length = decoder_layers
            noise_en = self.noise_en_fc(tf.random.normal(
                (self.encoder_layers, x_en.shape[0], x_en.shape[1], self.embed_dim), stddev=self.noise_std))
            noise_de = self.noise_de_fc(tf.random.normal(
                (self.encoder_layers, x_en.shape[0], 1, self.embed_dim), stddev=self.noise_std))
            # x_out: (batch, 1, embed_dim)
            x_out, _ = self.chords_extend((x_en, x_de, None, None), noise_en=noise_en, noise_de=noise_de)
            x_out = AdaInstanceNormalization()([x_out, beta, gamma])
            x_en = tf.concat((x_en, x_out), axis=1)  # (batch, en_time_in+1, embed_dim)
        return x_en[:, self.strt_dim:, :]  # (batch, out_seq_len, embed_dim)


class TimeSynthesis(tf.keras.models.Model):

    def __init__(self, time_features=3, init_knl=3, strt_dim=5, fc_activation=tf.keras.activations.relu,
                 encoder_layers=1, decoder_layers=1, fc_layers=3, norm_epsilon=1e-6, transformer_dropout_rate=0.2,
                 noise_std=0.5):
        super(TimeSynthesis, self).__init__()
        self.time_features = time_features
        self.strt_dim = strt_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.noise_std = noise_std
        self.embed_dim = time_features
        self.init_fc = tf.keras.layers.Dense(time_features)
        self.init_ext = tf.keras.layers.Conv1DTranspose(filters=time_features, kernel_size=init_knl, strides=strt_dim)
        self.time_extend = transformer.Transformer(
            embed_dim=time_features, n_heads=1, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            fc_layers=fc_layers, norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate,
            fc_activation=fc_activation)
        self.b_fc = tf.keras.layers.Dense(time_features, kernel_initializer='he_normal', bias_initializer='zeros')
        self.g_fc = tf.keras.layers.Dense(time_features, kernel_initializer='he_normal', bias_initializer='ones')
        self.noise_en_fc = tf.keras.layers.Dense(time_features, kernel_initializer='he_normal', bias_initializer='zeros')
        self.noise_de_fc = tf.keras.layers.Dense(time_features, kernel_initializer='he_normal', bias_initializer='zeros')

    def call(self, inputs, training=None, mask=None):
        # styl: (batch, in_dim)
        styl, out_seq_len = inputs
        x_en = self.init_fc(styl)  # (batch, time_features)
        x_en = self.init_ext(x_en[:, tf.newaxis, :])  # (batch, strt_dim, time_features)
        beta = self.b_fc(styl)[:, tf.newaxis, :]  # (batch, 1, time_features)
        gamma = self.g_fc(styl)[:, tf.newaxis, :]  # (batch, 1, time_features)
        x_out = self.extend_x(x_en, out_seq_len, beta, gamma)  # (batch, out_seq_len, time_features)
        return x_out

    def extend_x(self, x_en, out_seq_len, beta, gamma):
        # x_en: (batch, strt_dim, time_features)
        # beta: (batch, 1, time_features)
        # gamma: (batch, 1, time_features)
        x_de = tf.ones((x_en.shape[0], 1, self.time_features))  # (batch, 1, time_features)
        for i in range(out_seq_len):
            # noise_en: list of noise (batch, en_time_in, time_features), list length = encoder_layers
            # noise_de: list of noise (batch, out_seq_len, time_features), list length = decoder_layers
            noise_en = self.noise_en_fc(tf.random.normal(
                (self.encoder_layers, x_en.shape[0], x_en.shape[1], self.time_features), stddev=self.noise_std))
            noise_de = self.noise_de_fc(tf.random.normal(
                (self.encoder_layers, x_en.shape[0], 1, self.time_features), stddev=self.noise_std))
            # x_out: (batch, 1, time_features)
            x_out, _ = self.time_extend((x_en, x_de, None, None), noise_en=noise_en, noise_de=noise_de)
            x_out = AdaInstanceNormalization()([x_out, beta, gamma])
            x_en = tf.concat((x_en, x_out), axis=1)  # (batch, en_time_in+1, time_features)
        return x_en[:, self.strt_dim:, :]  # (batch, out_seq_len, time_features)
