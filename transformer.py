import tensorflow as tf


class MultiheadAttention(tf.keras.layers.Layer):

    def __init__(self, n_heads, embed_dim):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.depth = embed_dim // n_heads
        self.q_w = tf.keras.layers.Dense(embed_dim)
        self.k_w = tf.keras.layers.Dense(embed_dim)
        self.v_w = tf.keras.layers.Dense(embed_dim)

    def call(self, q, k, v, mask=None):
        query = self.q_w(q)  # (batch, time_out, embed_dim)
        key = self.k_w(k)  # (batch, time_in, embed_dim)
        value = self.v_w(v)  # (batch, time_in, embed_dim)

        # split heads
        query = MultiheadAttention.split_head(query, self.n_heads, self.depth)  # (batch, n_heads, time_out, depth)
        key = MultiheadAttention.split_head(key, self.n_heads, self.depth)  # (batch, n_heads, time_in, depth)
        value = MultiheadAttention.split_head(value, self.n_heads, self.depth)  # (batch, n_heads, time_in, depth)

        # self attention
        score = tf.matmul(query, key, transpose_b=True)  # (batch, n_heads, time_out, time_in)
        scaled_score = score / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))  # (batch, n_heads, time_out, time_in)
        if mask is not None:
            scaled_score += (mask * -1e9)
        weights = tf.nn.softmax(scaled_score, axis=-1)  # (batch, n_heads, length, length)
        attention = tf.matmul(weights, value)  # (batch, n_heads, length, depth)

        # combine heads
        context = tf.transpose(attention, [0, 2, 1, 3])  # (batch, length, n_heads, depth)
        context = tf.reshape(context, [tf.shape(context)[0], -1,
                                       context.shape[2] * attention.shape[3]])  # (batch_size, time_in, embed_dim)
        return context, weights

    @staticmethod
    def split_head(x, n_heads, depth):
        # x dim: (batch, time_in, embed_dim)
        x = tf.reshape(x, [tf.shape(x)[0], -1, n_heads, depth])  # (batch, length, n_heads, depth)
        x = tf.transpose(x, [0, 2, 1, 3])  # (batch, n_heads, length, depth)
        return x


class Transformer(tf.keras.Model):

    def __init__(self, embed_dim=256, n_heads=4, out_notes_pool_size=1000, encoder_layers=2, decoder_layers=2,
                 fc_layers=2, norm_epsilon=1e-6, dropout_rate=0.2, fc_activation="relu", type_='melody'):
        super(Transformer, self).__init__()
        assert embed_dim % n_heads == 0
        # self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.depth = embed_dim // n_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.fc_layers = fc_layers
        self.norm_epsilon = norm_epsilon
        self.dropout_rate = dropout_rate
        self.fc_activation = fc_activation

        # layers for encoder -------------------------
        self.mha_en = MultiheadAttention(n_heads, embed_dim)
        self.ln_en = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)
        self.fc_en = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(fc_layers, activation=fc_activation),
             tf.keras.layers.Dense(embed_dim),
             tf.keras.layers.Dropout(dropout_rate)])

        # layers for decoder -------------------------
        self.mha1_de = MultiheadAttention(n_heads, embed_dim)
        self.ln1_de = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)
        self.mha2_de = MultiheadAttention(n_heads, embed_dim)
        self.ln2_de = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)
        self.fc_de = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(fc_layers, activation=fc_activation),
             tf.keras.layers.Dense(embed_dim),
             tf.keras.layers.Dropout(dropout_rate)])
        self.ln3_de = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)

        # layer for output --------------------------
        self.final = tf.keras.layers.Dense(out_notes_pool_size) if type_ == 'melody' else tf.keras.layers.Dense(1)

    def call(self, x_en, x_de, mask_padding, mask_lookahead):
        # x_en: (batch, en_time_in, embed_dim)
        # x_de: (batch, de_time_in, embed_dim)
        # --------------------------- encoder ---------------------------
        x_en_ = x_en
        for i in range(self.encoder_layers):
            x_en_ = self.transformer_encoder_block(x_en=x_en_, mask_padding=mask_padding)

        # --------------------------- decoder ---------------------------
        all_weights = dict()
        x_de_ = x_de
        for i in range(self.decoder_layers):
            x_de_, all_weights['de_' + str(i + 1) + '_att_1'], all_weights['de_' + str(i + 1) + '_att_2'] = \
                Transformer.transformer_decoder_block(
                    x_de=x_de_, en_out=x_en_, mask_lookahead=mask_lookahead, mask_padding=mask_padding)

        # --------------------------- output ---------------------------
        # if type_ == 'melody': out: (batch, de_time_in, out_notes_pool_size)
        # else: out: (batch, de_time_in, 1)
        out = self.final(x_de_)
        return out, all_weights

    def transformer_encoder_block(self, x_en, mask_padding):
        # x_en dim: (batch, time_in, embed_dim)
        # --------------------------- sub-layer 1 ---------------------------
        # attention: (batch, time_in, embed_dim), att_weights: (batch, n_heads, length, length)
        attention, att_weights = self.mha_en(q=x_en, k=x_en, v=x_en, mask=mask_padding)
        attention = tf.keras.layers.Dropout(self.dropout_rate)(attention)  # (batch, time_in, embed_dim)
        skip_conn_1 = tf.keras.layers.Add()([x_en, attention])  # (batch, time_in, embed_dim)
        skip_conn_1 = self.ln_en(skip_conn_1)  # (batch, time_in, embed_dim)
        # --------------------------- sub-layer 2 ---------------------------
        fc = self.fc_en(skip_conn_1)  # (batch, time_in, embed_dim)
        skip_conn_2 = tf.keras.layers.Add()([skip_conn_1, fc])  # (batch, time_in, embed_dim)
        out = self.ln_en(skip_conn_2)  # (batch, time_in, embed_dim)
        return out

    def transformer_decoder_block(self, x_de, en_out, mask_lookahead, mask_padding):
        # x_de dim: (batch, de_time_in, embed_dim)
        # en_out dim: (batch, en_time_in, embed_dim)
        # --------------------------- sub-layer 1 ---------------------------
        # attention_1: (batch, de_time_in, embed_dim), att_weights_1: (batch, n_heads, length, length)
        attention_1, att_weights_1 = self.mha1_de(q=x_de, k=x_de, v=x_de, mask=mask_lookahead)
        attention_1 = tf.keras.layers.Dropout(self.dropout_rate)(attention_1)  # (batch, de_time_in, embed_dim)
        skip_conn_1 = tf.keras.layers.Add()([x_de, attention_1])  # (batch, de_time_in, embed_dim)
        skip_conn_1 = self.ln1_de(skip_conn_1)  # (batch, de_time_in, embed_dim)

        # --------------------------- sub-layer 2 ---------------------------
        # attention_2: (batch, time_in, embed_dim), att_weights_2: (batch, n_heads, length, length)
        # input tuple: (query: skip_conn_1, key: encoder_out, value: encoder_out)
        attention_2, att_weights_2 = self.mha2_de(q=skip_conn_1, k=en_out, v=en_out, mask=mask_padding)
        attention_2 = tf.keras.layers.Dropout(self.dropout_rate)(attention_2)  # (batch, de_time_in, embed_dim)
        skip_conn_2 = tf.keras.layers.Add()([attention_2, skip_conn_1])  # (batch, de_time_in, embed_dim)
        skip_conn_2 = self.ln2_de(skip_conn_2)  # (batch, de_time_in, embed_dim)

        # --------------------------- sub-layer 3 ---------------------------
        # fc: (batch, de_time_in, embed_dim)
        fc = self.fc_de(skip_conn_2)
        skip_conn_3 = tf.keras.layers.Add()([fc, skip_conn_2])  # (batch, de_time_in, embed_dim)
        out = self.ln3_de(skip_conn_3)  # (batch, de_time_in, embed_dim)
        return out, att_weights_1, att_weights_2


# todo: test  Transformer



