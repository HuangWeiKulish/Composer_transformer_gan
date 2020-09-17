import tensorflow as tf
import transformer


class ChordsDiscriminator(tf.keras.Model):

    def __init__(self, embed_dim=16, n_heads=4, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3,
                 norm_epsilon=1e-6, transformer_dropout_rate=0.2, pre_out_dim=512, out_dropout=0.3):
        super(ChordsDiscriminator, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'
        self.out_dropout = out_dropout
        self.discr = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_chords_pool_size=pre_out_dim,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)
        self.last_fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, return_vec=False, training=None, mask=None):
        nt_in, de_in = inputs
        # nt_in: (batch, seq_len, embed_dim)
        # de_in: (batch, 1, embed_dim): the '<start>' embedding
        pre_out, _ = self.discr((nt_in, de_in, None, None))  # pre_out: (batch, 1, pre_out_dim)
        out = self.last_fc(pre_out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Dropout(self.out_dropout)(out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        if return_vec:
            return pre_out, out
        return out


class TimeDiscriminator(tf.keras.Model):

    def __init__(self, time_features=3, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3,
                 norm_epsilon=1e-6, transformer_dropout_rate=0.2, pre_out_dim=512, out_dropout=0.3):
        super(TimeDiscriminator, self).__init__()
        self.out_dropout = out_dropout
        self.discr = transformer.Transformer(
            embed_dim=time_features, n_heads=1, out_chords_pool_size=pre_out_dim,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation,
            out_positive=False)
        self.last_fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, return_vec=False, training=None, mask=None):
        tm_in, de_in = inputs
        # de_in: (batch, 1, 3): the '<start>' embedding
        # tm_in: (batch, seq_len, 3): 3 for [velocity, velocity, time since last start, notes duration]
        pre_out, _ = self.discr((tm_in, de_in, None, None))  # pre_out: (batch, 1, pre_out_dim)
        out = self.last_fc(pre_out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Dropout(self.out_dropout)(out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        if return_vec:
            return pre_out, out
        return out


class Discriminator(tf.keras.Model):

    def __init__(self, embed_dim=16, n_heads=4, kernel_size=3, fc_activation="relu", encoder_layers=2, decoder_layers=2,
                 fc_layers=3, norm_epsilon=1e-6, transformer_dropout_rate=0.2, pre_out_dim=512, out_dropout=0.3):
        super(Discriminator, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'
        self.out_dropout = out_dropout
        self.tm_in_expand = tf.keras.layers.Conv1D(
            filters=embed_dim, kernel_size=kernel_size, strides=1, padding='same')
        self.discr = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_chords_pool_size=pre_out_dim,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)
        self.last_fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, return_vec=False, training=None, mask=None):
        nt_in, tm_in, de_in = inputs
        # de_in: (batch, 1, embed_dim): the '<start>' embedding
        # nt_in: (batch, seq_len, embed_dim)
        # tm_in: (batch, seq_len, 3): 3 for [velocity, velocity, time since last start, notes duration]
        tm_in_ = self.tm_in_expand(tm_in)  # (batch, seq_len, embed_dim)
        disr_in = tf.math.add(nt_in, tm_in_)  # (batch, seq_len, embed_dim)
        pre_out, _ = self.discr((disr_in, de_in, None, None))  # pre_out: (batch, 1, 1)
        out = self.last_fc(pre_out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Dropout(self.out_dropout)(out)  # out: (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        if return_vec:
            return pre_out, out
        return out

