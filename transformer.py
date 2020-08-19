import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Reshape, Add, Dropout, Embedding, Lambda


class TransformerBlocks:

    @staticmethod
    def transformer(x_en, x_de, mask_padding, mask_lookahead, out_notes_pool_size, embed_dim=256, encoder_layers=3,
                    decoder_layers=3, n_heads=2, depth=1, fc_layers=2, norm_epsilon=1e-6, transformer_dropout_rate=0.2,
                    fc_activation="relu", type='melody'):
        # x_en: (batch, en_time_in, embed_dim)
        # x_de: (batch, de_time_in, embed_dim)
        # --------------------------- encoder ---------------------------
        x_en_ = x_en
        for i in range(encoder_layers):
            x_en_ = TransformerBlocks.transformer_encoder_block(
                x_en=x_en_, mask_padding=mask_padding, embed_dim=embed_dim, n_heads=n_heads, depth=depth,
                fc_layers=fc_layers, norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate,
                fc_activation=fc_activation)

        # --------------------------- decoder ---------------------------
        all_weights = dict()
        x_de_ = x_de
        for i in range(decoder_layers):
            x_de_, all_weights['de_' + str(i + 1) + '_att_1'], all_weights['de_' + str(i + 1) + '_att_2'] = \
                TransformerBlocks.transformer_decoder_block(
                    x_de=x_de_, en_out=x_en_, mask_lookahead=mask_lookahead, mask_padding=mask_padding,
                    embed_dim=embed_dim, n_heads=n_heads, depth=depth, fc_layers=fc_layers, norm_epsilon=norm_epsilon,
                    dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)

        # --------------------------- output ---------------------------
        if type == 'melody':
            out = Dense(out_notes_pool_size)(x_de_)  # out: (batch, de_time_in, out_notes_pool_size)
        else:  # duration
            out = Dense(1)(x_de_)  # out: (batch, de_time_in, 1)
        return out, all_weights

    @staticmethod
    def split_head(x, n_heads=2, depth=1):
        # x dim: (batch, time_in, embed_dim)
        x = tf.reshape(x, [tf.shape(x)[0], -1, n_heads, depth])  # (batch, length, n_heads, depth)
        x = tf.transpose(x, [0, 2, 1, 3])  # (batch, n_heads, length, depth)
        return x

    @staticmethod
    def multi_head_self_attention(q, k, v, mask=None, embed_dim=256, n_heads=2, depth=1):
        query = Dense(embed_dim)(q)  # (batch, time_out, embed_dim)
        key = Dense(embed_dim)(k)  # (batch, time_in, embed_dim)
        value = Dense(embed_dim)(v)  # (batch, time_in, embed_dim)

        # split heads
        query = TransformerBlocks.split_head(query, n_heads, depth)  # (batch, n_heads, time_out, depth)
        key = TransformerBlocks.split_head(key, n_heads, depth)  # (batch, n_heads, time_in, depth)
        value = TransformerBlocks.split_head(value, n_heads, depth)  # (batch, n_heads, time_in, depth)

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
    def feed_forward(x, embed_dim=1, fc_layers=3, dropout_rate=0.2, activation="relu"):
        fc = Dense(fc_layers, activation=activation)(x)  # (batch, time_in, fc_layers)
        fc = Dense(embed_dim)(fc)  # (batch, time_in, embed_dim)
        fc = Dropout(dropout_rate)(fc)  # (batch, time_in, embed_dim)
        return fc

    @staticmethod
    def transformer_encoder_block(x_en, mask_padding, embed_dim=256, n_heads=4, depth=1, fc_layers=3,
                                  norm_epsilon=1e-6, dropout_rate=0.2, fc_activation="relu"):
        # x dim: (batch, time_in, embed_dim)
        # --------------------------- sub-layer 1 ---------------------------
        # attention: (batch, time_in, embed_dim), att_weights: (batch, n_heads, length, length)
        attention, att_weights = TransformerBlocks.multi_head_self_attention(
            q=x_en, k=x_en, v=x_en, mask=mask_padding, embed_dim=embed_dim, n_heads=n_heads, depth=depth)
        attention = Dropout(dropout_rate)(attention)  # (batch, time_in, embed_dim)
        skip_conn_1 = Add()([x_en, attention])  # (batch, time_in, embed_dim)
        skip_conn_1 = LayerNormalization(epsilon=norm_epsilon)(skip_conn_1)  # (batch, time_in, embed_dim)

        # --------------------------- sub-layer 2 ---------------------------
        fc = TransformerBlocks.feed_forward(
            skip_conn_1, embed_dim, fc_layers, dropout_rate, fc_activation)  # (batch, time_in, embed_dim)
        skip_conn_2 = Add()([skip_conn_1, fc])  # (batch, time_in, embed_dim)
        out = LayerNormalization(epsilon=norm_epsilon)(skip_conn_2)  # (batch, time_in, embed_dim)
        return out

    @staticmethod
    def transformer_decoder_block(x_de, en_out, mask_lookahead, mask_padding, embed_dim=256, n_heads=2, depth=1,
                                  fc_layers=3, norm_epsilon=1e-6, dropout_rate=0.2, fc_activation="relu"):
        # x_de dim: (batch, de_time_in, embed_dim)
        # en_out dim: (batch, en_time_in, embed_dim)
        # --------------------------- sub-layer 1 ---------------------------
        # attention_1: (batch, de_time_in, embed_dim), att_weights_1: (batch, n_heads, length, length)
        attention_1, att_weights_1 = TransformerBlocks.multi_head_self_attention(
            q=x_de, k=x_de, v=x_de, mask=mask_lookahead, embed_dim=embed_dim, n_heads=n_heads, depth=depth)
        attention_1 = Dropout(dropout_rate)(attention_1)  # (batch, de_time_in, embed_dim)
        skip_conn_1 = Add()([x_de, attention_1])  # (batch, de_time_in, embed_dim)
        skip_conn_1 = LayerNormalization(epsilon=norm_epsilon)(skip_conn_1)  # (batch, de_time_in, embed_dim)

        # --------------------------- sub-layer 2 ---------------------------
        # attention_2: (batch, time_in, embed_dim), att_weights_2: (batch, n_heads, length, length)
        # input tuple: (query: skip_conn_1, key: encoder_out, value: encoder_out)
        attention_2, att_weights_2 = TransformerBlocks.multi_head_self_attention(
            q=skip_conn_1, k=en_out, v=en_out, mask=mask_padding, embed_dim=embed_dim, n_heads=n_heads, depth=depth)
        attention_2 = Dropout(dropout_rate)(attention_2)  # (batch, de_time_in, embed_dim)
        skip_conn_2 = Add()([attention_2, skip_conn_1])  # (batch, de_time_in, embed_dim)
        skip_conn_2 = LayerNormalization(epsilon=norm_epsilon)(skip_conn_2)  # (batch, de_time_in, embed_dim)

        # --------------------------- sub-layer 3 ---------------------------
        # fc: (batch, de_time_in, embed_dim)
        fc = TransformerBlocks.feed_forward(skip_conn_2, embed_dim, fc_layers, dropout_rate, fc_activation)
        skip_conn_3 = Add()([fc, skip_conn_2])  # (batch, de_time_in, embed_dim)
        out = LayerNormalization(epsilon=norm_epsilon)(skip_conn_3)  # (batch, de_time_in, embed_dim)
        return out, att_weights_1, att_weights_2

