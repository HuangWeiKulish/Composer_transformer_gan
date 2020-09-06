import tensorflow as tf
import util, transformer, generator

notes_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_embedder'
notes_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_generator'
time_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_generator'


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
        self.fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, nt_in, tm_in, de_in, mask_padding, mask_lookahead):
        # nt_in: (batch, seq_len, embed_dim)
        # tm_in: (batch, seq_len, 3): 3 for [velocity, velocity, time since last start, notes duration]
        # de_in: (batch, 1, embed_dim): the '<start>' embedding
        # mask_padding = None  # util.padding_mask(x_in[:, :, 0])  # (batch, 1, 1, in_seq_len)
        # mask_lookahead = util.lookahead_mask(nt_in.shape[1])  # (out_seq_len, out_seq_len)
        tm_in_ = self.tm_in_expand(tm_in)  # (batch, seq_len, embed_dim)
        combined = tf.math.add(nt_in, tm_in_)  # (batch, seq_len, embed_dim)
        out, _ = self.discr(combined, de_in, mask_padding, mask_lookahead)  # out: (batch, 1, 1)
        out = self.fc(out)  # (batch, 1, 1)
        out = tf.keras.layers.Flatten()(out)  # (batch, 1)
        return out
