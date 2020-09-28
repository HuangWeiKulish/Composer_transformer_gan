import tensorflow as tf
import transformer


class ChordsDiscriminator(tf.keras.Model):

    def __init__(self, embed_dim=16, n_heads=4, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3,
                 norm_epsilon=1e-6, transformer_dropout_rate=0.2, pre_out_dim=512, out_dropout=0.3,
                 recycle_fc_activ=tf.keras.activations.elu):
        super(ChordsDiscriminator, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'
        self.out_dropout = out_dropout
        self.embed_dim = embed_dim
        self.discr = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            fc_layers=fc_layers, norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate,
            fc_activation=fc_activation)
        self.recycle_fc = tf.keras.layers.Dense(pre_out_dim, activation=recycle_fc_activ)
        self.last_fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, ch_in, training=None, mask=None):
        # ch_in: (batch, seq_len, embed_dim)
        de_in = tf.ones((ch_in.shape[0], 1, self.embed_dim))  # (batch, 1, embed_dim)
        pre_out, _ = self.discr((ch_in, de_in, None, None), noise_en=None, noise_de=None)  # (batch, 1, embed_dim)
        pre_out = self.recycle_fc(pre_out)  # (batch, 1, pre_out_dim)
        out = self.last_fc(pre_out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Dropout(self.out_dropout)(out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        return tf.squeeze(pre_out, axis=1), out


class TimeDiscriminator(tf.keras.Model):

    def __init__(self, time_features=3, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3,
                 norm_epsilon=1e-6, transformer_dropout_rate=0.2, pre_out_dim=512, out_dropout=0.3,
                 recycle_fc_activ=tf.keras.activations.elu):
        super(TimeDiscriminator, self).__init__()
        self.time_features = time_features
        self.out_dropout = out_dropout
        self.discr = transformer.Transformer(
            embed_dim=time_features, n_heads=1, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            fc_layers=fc_layers, norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate,
            fc_activation=fc_activation)
        self.recycle_fc = tf.keras.layers.Dense(pre_out_dim, activation=recycle_fc_activ)
        self.last_fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, tm_in, training=None, mask=None):
        # tm_in: (batch, seq_len, 3): 3 for [velocity, velocity, time since last start, notes duration]
        de_in = tf.zeros((tm_in.shape[0], 1, self.time_features))  # (batch, 1, time_features)
        pre_out, _ = self.discr((tm_in, de_in, None, None), noise_en=None, noise_de=None)  # (batch, 1, time_features)
        pre_out = self.recycle_fc(pre_out)  # out: (batch, 1, pre_out_dim)
        out = self.last_fc(pre_out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Dropout(self.out_dropout)(out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        return tf.squeeze(pre_out, axis=1), out


class Discriminator(tf.keras.Model):

    def __init__(self, embed_dim=16, n_heads=4, kernel_size=3, fc_activation="relu", encoder_layers=2, decoder_layers=2,
                 fc_layers=3, norm_epsilon=1e-6, transformer_dropout_rate=0.2, pre_out_dim=512, out_dropout=0.3,
                 recycle_fc_activ=tf.keras.activations.elu):
        super(Discriminator, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'
        self.out_dropout = out_dropout
        self.embed_dim = embed_dim
        self.tm_in_expand = tf.keras.layers.Conv1D(
            filters=embed_dim, kernel_size=kernel_size, strides=1, padding='same')
        self.discr = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            fc_layers=fc_layers, norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate,
            fc_activation=fc_activation)
        self.comb_fc = tf.keras.layers.Dense(1)
        self.recycle_fc = tf.keras.layers.Dense(pre_out_dim, activation=recycle_fc_activ)
        self.last_fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        ch_in, tm_in = inputs
        # de_in: (batch, 1, embed_dim): the '<start>' embedding
        # ch_in: (batch, seq_len, embed_dim)
        # tm_in: (batch, seq_len, 3): 3 for [velocity, velocity, time since last start, notes duration]
        tm_in_ = self.tm_in_expand(tm_in)  # (batch, seq_len, embed_dim)
        disr_in = self.comb_fc(tf.concat(
            [ch_in[:, :, :, tf.newaxis], tm_in_[:, :, :, tf.newaxis]], axis=-1))  # (batch, seq_len, embed_dim, 1)
        disr_in = tf.squeeze(disr_in, axis=-1)  # (batch, seq_len, embed_dim)
        de_in = tf.ones((ch_in.shape[0], 1, self.embed_dim))  # (batch, 1, embed_dim)
        pre_out, _ = self.discr((disr_in, de_in, None, None), noise_en=None, noise_de=None)  # (batch, 1, embed_dim)
        pre_out = self.recycle_fc(pre_out)  # (batch, 1, pre_out_dim)
        out = self.last_fc(pre_out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Dropout(self.out_dropout)(out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        return tf.squeeze(pre_out, axis=1), out
