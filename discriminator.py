import tensorflow as tf
import transformer


class NotesDiscriminator(tf.keras.Model):

    def __init__(self, embed_dim=256, n_heads=4, fc_activation="relu", encoder_layers=2,
                 decoder_layers=2, fc_layers=3, norm_epsilon=1e-6, transformer_dropout_rate=0.2):
        super(NotesDiscriminator, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'
        self.discr = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=1,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)

    def call(self, nt_in, de_in):
        # nt_in: (batch, seq_len, embed_dim)
        # de_in: (batch, 1, embed_dim): the '<start>' embedding
        out, _ = self.discr(nt_in, de_in, mask_padding=None, mask_lookahead=None)  # out: (batch, 1, 1)
        out = tf.keras.layers.Activation('sigmoid')(out)  # (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        return out


class TimeDiscriminator(tf.keras.Model):

    def __init__(self, time_features=3, fc_activation="relu", encoder_layers=2,
                 decoder_layers=2, fc_layers=3, norm_epsilon=1e-6, transformer_dropout_rate=0.2):
        super(TimeDiscriminator, self).__init__()
        self.discr = transformer.Transformer(
            embed_dim=time_features, n_heads=1, out_notes_pool_size=1,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)

    def call(self, tm_in, de_in):
        # de_in: (batch, 1, 3): the '<start>' embedding
        # tm_in: (batch, seq_len, 3): 3 for [velocity, velocity, time since last start, notes duration]
        out, _ = self.discr(tm_in, de_in, mask_padding=None, mask_lookahead=None)  # out: (batch, 1, 1)
        out = tf.keras.layers.Activation('sigmoid')(out)  # (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        return out


class Discriminator(tf.keras.Model):

    def __init__(self, embed_dim=256, n_heads=4, kernel_size=3, fc_activation="relu", encoder_layers=2,
                 decoder_layers=2, fc_layers=3, norm_epsilon=1e-6, transformer_dropout_rate=0.2):
        super(Discriminator, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'
        self.tm_in_expand = tf.keras.layers.Conv1D(
            filters=embed_dim, kernel_size=kernel_size, strides=1, padding='same')
        self.discr = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=1,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)

    def call(self, nt_in, tm_in, de_in):
        # de_in: (batch, 1, embed_dim): the '<start>' embedding
        # nt_in: (batch, seq_len, embed_dim)
        # tm_in: (batch, seq_len, 3): 3 for [velocity, velocity, time since last start, notes duration]
        tm_in_ = self.tm_in_expand(tm_in)  # (batch, seq_len, embed_dim)
        disr_in = tf.math.add(nt_in, tm_in_)  # (batch, seq_len, embed_dim)
        out, _ = self.discr(disr_in, de_in, mask_padding=None, mask_lookahead=None)  # out: (batch, 1, 1)
        out = tf.keras.layers.Activation('sigmoid')(out)  # (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        return out
