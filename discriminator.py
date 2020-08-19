import tensorflow as tf
from tensorflow.keras.layers import Dense
import util, transformer, generator


class Discriminator(tf.keras.Model):

    def __init__(self, notes_pool_size, max_pos, embed_dim, dropout_rate=0.2, n_heads=4, fc_layers=3,
                 norm_epsilon=1e-6, fc_activation="relu", encoder_layers=3, decoder_layers=3, ):
        super(Discriminator, self).__init__()
        self.embedder = generator.Embedder(notes_pool_size, max_pos, embed_dim, dropout_rate)
        self.notes_pool_size = notes_pool_size
        self.n_heads = n_heads
        self.depth = embed_dim // n_heads
        self.fc_layers = fc_layers
        self.norm_epsilon = norm_epsilon
        self.fc_activation = fc_activation
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

    def call(self, x_out_melody, x_out_duration, x_de_in, mask_padding, mask_lookahead):
        # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
        # x_out_duration: (batch, out_seq_len, 1)
        # x_de_in: (batch, 2): 2 columns: [tk.word_index['<start>'], 0]
        melody_ = util.softargmax(x_out_melody, beta=1e10)  # return the index of the max value: (batch, seq_len)
        melody_ = tf.expand_dims(melody_, axis=-1)  # (batch, seq_len, 1)
        music_ = tf.concat([melody_, x_out_duration], axis=-1)  # (batch, seq_len, 2): column: notes_id, duration
        emb = self.embedder(music_)  # (batch, seq_len, embed_dim)

        x_de_in = tf.expand_dims(x_de_in, 1)  # (batch, 1, 2)
        x_de = self.embedder(x_de_in)  # (batch, 1, embed_dim)

        # (batch, out_seq_len, 1)
        out, _ = transformer.TransformerBlocks.transformer(
            emb, x_de, mask_padding, mask_lookahead, out_notes_pool_size=self.notes_pool_size, embed_dim=self.embed_dim,
            encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers, n_heads=self.n_heads,
            depth=self.depth, fc_layers=self.fc_layers, norm_epsilon=self.norm_epsilon,
            transformer_dropout_rate=self.transformer_dropout_rate,
            fc_activation=self.fc_activation, type='')  # (None, None, 1)

        """
        out, _ = TransformerBlocks.transformer(
            emb, x_de, mask_padding, mask_lookahead, out_notes_pool_size=notes_pool_size, embed_dim=embed_dim,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, n_heads=n_heads, depth=depth,
            fc_layers=fc_layers, norm_epsilon=norm_epsilon,
            transformer_dropout_rate=transformer_dropout_rate,
            fc_activation=fc_activation, type='')
        """
        out = Dense(1, activation='sigmoid')(out)  # (None, None, 1)
        return out
