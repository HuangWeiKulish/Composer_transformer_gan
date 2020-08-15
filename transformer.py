import os
import glob
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import pickle as pkl
from scipy import sparse

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, BatchNormalization, LayerNormalization, GaussianNoise, \
    Flatten, Reshape, Activation, Conv1DTranspose, GRU, RepeatVector, Dot, TimeDistributed, concatenate, \
    Bidirectional, Add, Permute, Dropout, Embedding, Lambda
from tensorflow.keras.models import Model
from Side_Project.Composer.transformer_gan import util


class Embedder(tf.keras.Model):

    def __init__(self, notes_pool_size, max_pos, embed_dim, dropout_rate=0.2):
        super(Embedder, self).__init__()
        self.notes_pool_size = notes_pool_size
        self.max_pos = max_pos
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

    def call(self, x_in):
        # x_in: (batch, time_in, 2): the second dimension contains: notes, duration
        x_out = self.embedding(x_in, self.notes_pool_size, self.max_pos, self.embed_dim, self.dropout_rate)
        return x_out

    @staticmethod
    def get_angles(pos, i, embed_dim):
        # reference: https://www.tensorflow.org/tutorials/text/transformer
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(max_pos, embed_dim):
        # reference: https://www.tensorflow.org/tutorials/text/transformer
        angle_rads = Embedder.get_angles(
            np.arange(max_pos)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    @staticmethod
    def embedding(x, notes_pool_size, max_pos, embed_dim, dropout_rate=0.2):
        # x dim: (batch, time_in, 2)
        # --------------------------- split x into x_notes and x_duration ---------------------------
        x_notes = Lambda(lambda x_: x_[:, :, 0])(x)  # (batch, time_in), the string encoding of notes
        x_duration = Lambda(lambda x_: x_[:, :, 1])(x)  # (batch, time_in), the normalized duration
        # --------------------------- embedding ---------------------------
        notes_emb = Embedding(notes_pool_size, embed_dim)(x_notes)  # (batch, time_in, embed_dim)
        pos_emb = Embedder.positional_encoding(max_pos, embed_dim)  # (1, max_pos, embed_dim)
        # --------------------------- combine ---------------------------
        combine = notes_emb + pos_emb[:, :x.shape[1], :] + Reshape((x.shape[1], 1))(x_duration)
        if dropout_rate is not None:
            combine = Dropout(dropout_rate)(combine)
        return combine  # (batch, time_in, embed_dim)


class Generator(tf.keras.Model):

    def __init__(self, de_max_pos=10000, embed_dim=256, n_heads=4,
                 out_notes_pool_size=8000, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3,
                 norm_epsilon=1e-6, transformer_dropout_rate=0.2, embedding_dropout_rate=0.2):
        super(Generator, self).__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.out_notes_pool_size = out_notes_pool_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.n_heads = n_heads
        self.depth = embed_dim // n_heads
        self.fc_layers = fc_layers
        self.norm_epsilon = norm_epsilon
        self.transformer_dropout_rate = transformer_dropout_rate
        self.fc_activation = fc_activation
        self.embedder_de = Embedder(notes_pool_size=out_notes_pool_size, max_pos=de_max_pos,
                                    embed_dim=embed_dim, dropout_rate=embedding_dropout_rate)

    def call(self, x_en_melody, x_de_in, mask_padding_en, mask_padding_de, mask_lookahead):
        # x_en  # (batch, in_seq_len, embed_dim): x_en_in is the embed output of notes
        # x_de_in  # (batch, out_seq_len, 2): x_de_in is the un-embedded target

        # transformer for melody -----------------------------------------
        # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
        x_de_melody = self.embedder_de(x_de_in)  # (batch, out_seq_len, embed_dim)
        x_out_melody, all_weights_melody = Generator.transformer(
            x_en=x_en_melody, x_de=x_de_melody, mask_padding_en=mask_padding_en, mask_lookahead=mask_lookahead,
            mask_padding_de=mask_padding_de, out_notes_pool_size=self.out_notes_pool_size, embed_dim=self.embed_dim,
            encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers, n_heads=self.n_heads,
            depth=self.depth, fc_layers=self.fc_layers, norm_epsilon=self.norm_epsilon,
            transformer_dropout_rate=self.transformer_dropout_rate, fc_activation=self.fc_activation, type='melody')

        # transformer for duration -----------------------------------------
        # x_out_duration: (batch, out_seq_len, 1)
        x_de_duration = tf.slice(x_de_in, [0, 0, 1], [tf.shape(x_de_in)[0], tf.shape(x_de_in)[1], 1])  # (batch, out_seq_len, 1)
        x_out_duration, all_weights_duration = Generator.transformer(
            x_en=x_en_melody, x_de=x_de_duration, mask_padding_en=mask_padding_en, mask_lookahead=mask_lookahead,
            mask_padding_de=mask_padding_de, out_notes_pool_size=self.out_notes_pool_size, embed_dim=1,
            encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers, n_heads=1,
            depth=1, fc_layers=self.fc_layers, norm_epsilon=self.norm_epsilon,
            transformer_dropout_rate=self.transformer_dropout_rate, fc_activation=self.fc_activation, type='duration')

        return x_out_melody, all_weights_melody, x_out_duration, all_weights_duration

    @staticmethod
    def transformer(x_en, x_de, mask_padding_en, mask_lookahead, mask_padding_de, out_notes_pool_size,
                    embed_dim=256, encoder_layers=3, decoder_layers=3, n_heads=2, depth=1, fc_layers=2,
                    norm_epsilon=1e-6, transformer_dropout_rate=0.2, fc_activation="relu", type='melody'):
        # x_en: (batch, en_time_in, embed_dim)
        # x_de: (batch, de_time_in, embed_dim)
        # --------------------------- encoder ---------------------------
        x_en_ = x_en
        for i in range(encoder_layers):
            x_en_ = Generator.transformer_encoder_block(
                x_en=x_en_, mask_padding_en=mask_padding_en, embed_dim=embed_dim, n_heads=n_heads, depth=depth,
                fc_layers=fc_layers, norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate,
                fc_activation=fc_activation)

        # --------------------------- decoder ---------------------------
        all_weights = dict()
        x_de_ = x_de
        for i in range(decoder_layers):
            x_de_, all_weights['de_' + str(i + 1) + '_att_1'], all_weights['de_' + str(i + 1) + '_att_2'] = \
                Generator.transformer_decoder_block(
                    x_de=x_de_, en_out=x_en_, mask_lookahead=mask_lookahead, mask_padding=mask_padding_de,
                    embed_dim=embed_dim, n_heads=n_heads, depth=depth, fc_layers=fc_layers,
                    norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)

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
        query = Generator.split_head(query, n_heads, depth)  # (batch, n_heads, time_out, depth)
        key = Generator.split_head(key, n_heads, depth)  # (batch, n_heads, time_in, depth)
        value = Generator.split_head(value, n_heads, depth)  # (batch, n_heads, time_in, depth)

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
    def transformer_encoder_block(x_en, mask_padding_en, embed_dim=256, n_heads=2, depth=1, fc_layers=3,
                                  norm_epsilon=1e-6, dropout_rate=0.2, fc_activation="relu"):
        # x dim: (batch, time_in, embed_dim)
        # --------------------------- sub-layer 1 ---------------------------
        # attention: (batch, time_in, embed_dim), att_weights: (batch, n_heads, length, length)
        attention, att_weights = Generator.multi_head_self_attention(
            q=x_en, k=x_en, v=x_en, mask=mask_padding_en, embed_dim=embed_dim, n_heads=n_heads, depth=depth)
        attention = Dropout(dropout_rate)(attention)  # (batch, time_in, embed_dim)
        skip_conn_1 = Add()([x_en, attention])  # (batch, time_in, embed_dim)
        skip_conn_1 = LayerNormalization(epsilon=norm_epsilon)(skip_conn_1)  # (batch, time_in, embed_dim)

        # --------------------------- sub-layer 2 ---------------------------
        fc = Generator.feed_forward(skip_conn_1, embed_dim, fc_layers, dropout_rate, fc_activation)  # (batch, time_in, embed_dim)
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
        attention_1, att_weights_1 = Generator.multi_head_self_attention(
            q=x_de, k=x_de, v=x_de, mask=mask_lookahead, embed_dim=embed_dim, n_heads=n_heads, depth=depth)
        attention_1 = Dropout(dropout_rate)(attention_1)  # (batch, de_time_in, embed_dim)
        skip_conn_1 = Add()([x_de, attention_1])  # (batch, de_time_in, embed_dim)
        skip_conn_1 = LayerNormalization(epsilon=norm_epsilon)(skip_conn_1)  # (batch, de_time_in, embed_dim)

        # --------------------------- sub-layer 2 ---------------------------
        # attention_2: (batch, time_in, embed_dim), att_weights_2: (batch, n_heads, length, length)
        # input tuple: (query: skip_conn_1, key: encoder_out, value: encoder_out)
        attention_2, att_weights_2 = Generator.multi_head_self_attention(
            q=skip_conn_1, k=en_out, v=en_out, mask=mask_padding, embed_dim=embed_dim, n_heads=n_heads, depth=depth)
        attention_2 = Dropout(dropout_rate)(attention_2)  # (batch, de_time_in, embed_dim)
        skip_conn_2 = Add()([attention_2, skip_conn_1])  # (batch, de_time_in, embed_dim)
        skip_conn_2 = LayerNormalization(epsilon=norm_epsilon)(skip_conn_2)  # (batch, de_time_in, embed_dim)

        # --------------------------- sub-layer 3 ---------------------------
        # fc: (batch, de_time_in, embed_dim)
        fc = Generator.feed_forward(skip_conn_2, embed_dim, fc_layers, dropout_rate, fc_activation)
        skip_conn_3 = Add()([fc, skip_conn_2])  # (batch, de_time_in, embed_dim)
        out = LayerNormalization(epsilon=norm_epsilon)(skip_conn_3)  # (batch, de_time_in, embed_dim)
        return out, att_weights_1, att_weights_2


class GeneratorPretrain(tf.keras.Model):

    def __init__(self, en_max_pos=5000, de_max_pos=10000, embed_dim=256, n_heads=4, in_notes_pool_size=150000,
                 out_notes_pool_size=150000, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3,
                 norm_epsilon=1e-6, transformer_dropout_rate=0.2, embedding_dropout_rate=0.2, beta_1=0.9, beta_2=0.98,
                 epsilon=1e-9):
        super(GeneratorPretrain, self).__init__()
        self.embedder_en = Embedder(
            notes_pool_size=in_notes_pool_size, max_pos=en_max_pos, embed_dim=embed_dim,
            dropout_rate=embedding_dropout_rate)
        self.generator = Generator(
            de_max_pos=de_max_pos, embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=out_notes_pool_size,
            fc_activation=fc_activation, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            fc_layers=fc_layers, norm_epsilon=norm_epsilon, transformer_dropout_rate=transformer_dropout_rate,
            embedding_dropout_rate=embedding_dropout_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.learning_rate = util.CustomSchedule(embed_dim)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    def train_step(self, x_en_in, x_de_in):
        # x_en_in: (batch, in_seq_len, 2)
        # x_de_in: (batch, out_seq_len, 2)
        with tf.GradientTape() as tape:
            x_en_ = self.embedder_en(x_en_in)  # (batch, in_seq_len, embed_dim)
            mask_padding_en = util.padding_mask(x_en_in[:, :, 0])  # (batch, 1, 1, seq_len)
            mask_padding_de = util.padding_mask(x_en_in[:, :, 0])  # (batch, 1, 1, seq_len)
            mask_lookahead = util.lookahead_mask(x_de_in.shape[1])  # (seq_len, seq_len)

            # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
            # x_out_duration: (batch, out_seq_len, 1)
            x_out_melody, all_weights_melody, x_out_duration, all_weights_duration = self.generator(
                x_en_, x_de_in, mask_padding_en, mask_padding_de, mask_lookahead)







            loss = util.loss_func(x_de_in, x_de_out)

        variables = self.embedder_en.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        # self.train_loss(loss)
        # self.train_accuracy(tar_real, predictions)
        return loss

    def train(self, epochs):
        for epoch in range(epochs):
            start = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, self.train_loss.result(), self.train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def predict(self, x_in, tk, out_seq_len, dur_denorm=20):
        # x_in: (batch, in_seq_len, 2), 2 columns: [notes in text format, duration in integer format]
        batch_size = x_in.shape[0]

        x_in_ = util.number_encode_text(x_in, tk).astype(np.float32)
        x_en_ = self.embedder_en(x_in_)  # (batch, in_seq_len, embed_dim)

        # init x_de_in
        x_de_in = tf.constant([[tk.word_index['<start>'], 0]] * batch_size, dtype=tf.float32)
        x_de_in = tf.expand_dims(x_de_in, 1)  # (batch, 1, 2)

        result = []  # (out_seq_len, batch, 2)
        for i in range(out_seq_len):
            mask_padding_en = util.padding_mask(x_in_[:, :, 0])
            mask_padding_de = util.padding_mask(x_in_[:, :, 0])
            mask_lookahead = util.lookahead_mask(x_de_in.shape[1])

            # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
            # x_out_duration: (batch, out_seq_len, 1)
            x_out_melody, all_weights_melody, x_out_duration, all_weights_duration = self.generator(
                x_en_, x_de_in, mask_padding_en, mask_padding_de, mask_lookahead)

            notes_id = tf.expand_dims(tf.argmax(x_out_melody, -1)[:, -1], axis=1)
            notes_id = [nid for nid in notes_id.numpy()[:, 0]]  # len = batch
            durations = [dur for dur in x_out_duration.numpy()[:, 0, 0]]  # len = batch

            x_de_in_next = [[nid, dur] for nid, dur in zip(notes_id, durations)]
            x_de_in = tf.concat((x_de_in, tf.constant(x_de_in_next, dtype=tf.float32)[:, tf.newaxis, :]), axis=1)

            notes_dur = [[util.inds2notes(tk, nid, default='p'), dur * dur_denorm] for nid, dur in
                         zip(notes_id, durations)]  # [[notes1, dur1], ...] for all batches
            result.append(notes_dur)

        result = np.transpose(np.array(result, dtype=object), (1, 0, 2))  # (batch, out_seq_len, 2)
        return result









cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')  # from_logits=False

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
checkpoint_path = "./checkpoints/train"



# input_vocab_size = tokenizer_pt.vocab_size + 2
# target_vocab_size = tokenizer_en.vocab_size + 2
"""
in_seq_len=32
out_seq_len=128
lr=0.001
clipnorm=1.0
model_path_name=None
load_trainable=True
custom_loss=util.loss_func()
print_model=True
"""