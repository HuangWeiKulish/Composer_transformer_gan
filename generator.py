import numpy as np
import time
import tensorflow as tf
import util
import transformer

tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

chords_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_embedder'
chords_extend_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_extender'
time_extend_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_extender'

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

# tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_indexcer/chords_dict_final.pkl'
# tk = pkl.load(open(tk_path, 'rb'))


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

    def __init__(self, fc_layers=4, activ=tf.keras.layers.LeakyReLU(alpha=0.1)):
        super(Mapping, self).__init__()
        self.fcs = tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation=activ) for i in range(fc_layers)])

    def call(self, x, training=None):
        # x: (batch, in_dim, 1)
        mean_ = tf.math.reduce_mean(x, axis=[1])
        std_ = tf.math.reduce_std(x, axis=[1]) + tf.constant([0.0001], dtype=tf.float32)
        out = (x - mean_[:, :, np.newaxis]) / std_[:, :, np.newaxis]
        out = self.fcs(out, training=training)  # (batch, in_dim, 1)
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
        # x_in dim: (batch, time_in): chords index
        # --------------------------- split x into x_chords and x_duration ---------------------------
        # x_chords = tf.keras.layers.Lambda(lambda x_: x_[:, :, 0])(x_in)  # (batch, time_in), the string encoding of chords
        # --------------------------- embedding ---------------------------
        chords_emb = self.embedder(x_in)  # (batch, time_in, embed_dim)
        pos_emb = ChordsEmbedder.positional_encoding(self.max_pos, self.embed_dim)  # (1, max_pos, embed_dim)
        # --------------------------- combine ---------------------------
        combine = chords_emb + pos_emb[:, : x_in.get_shape()[1], :]  # (batch, time_in, embed_dim)
        if self.dropout_rate is not None:
            combine = tf.keras.layers.Dropout(self.dropout_rate)(combine)
        return combine  # (batch, time_in, embed_dim)

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


class ChordsExtendPretrain(tf.keras.models.Model):

    def __init__(self, strt_token_id=15001, out_chords_pool_size=15002, embed_dim=16, n_heads=4, max_pos=800,
                 fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3, norm_epsilon=1e-6,
                 embedding_dropout_rate=0.2, transformer_dropout_rate=0.2):
        super(ChordsExtendPretrain, self).__init__()
        assert embed_dim % n_heads == 0, 'make sure: embed_dim % n_heads == 0'
        self.strt_token_id = strt_token_id  # tk.word_index['<start>']
        self.embed_dim = embed_dim
        self.chords_emb = ChordsEmbedder(
            chords_pool_size=out_chords_pool_size, max_pos=max_pos, embed_dim=embed_dim,
            dropout_rate=embedding_dropout_rate)
        self.chords_extend = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_chords_pool_size=out_chords_pool_size,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation,
            out_positive=False)

    def call(self, inputs, training=None, mask=None):
        # x_en: (batch_size, in_seq_len, embed_dim)
        # x_de: (batch_size, 1, embed_dim)
        x_en, out_seq_len = inputs
        batch_size = x_en.shape[0]
        x_de = tf.constant([[self.strt_token_id]] * batch_size, dtype=tf.float32)
        x_de = self.chords_emb(x_de)
        for i in range(out_seq_len):
            mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
            # x_out: (batch_size, 1, out_chords_pool_size)
            x_out, _ = self.chords_extend((x_en, x_de, mask_padding, mask_lookahead))
            # translate prediction to text
            pred = tf.argmax(x_out, -1)  # (batch_size, 1)
            x_de = tf.concat((x_de, self.chords_emb(pred[:, -1][:, tf.newaxis])), axis=1)
        return x_de[:, 1:, :]  # (batch_size, out_seq_len, embed_dim)

    def predict_chords(self, x_en, tk, out_seq_len, return_str=True):
        # x_en: (batch_size, in_seq_len, embed_dim)
        # x_de: (batch_size, 1, embed_dim)
        batch_size = x_en.shape[0]
        x_de = tf.constant([[tk.word_index['<start>']]] * batch_size, dtype=tf.float32)
        x_de = self.chords_emb(x_de)
        result = []  # (out_seq_len, batch)
        for i in range(out_seq_len):
            mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
            # x_out: (batch_size, 1, out_chords_pool_size)
            x_out, _ = self.chords_extend((x_en, x_de, mask_padding, mask_lookahead))
            # translate prediction to text
            pred = tf.argmax(x_out, -1)
            pred = [nid for nid in pred.numpy()[:, -1]]  # len = batch, take the last prediction
            # append pred to x_de:
            x_de = tf.concat((x_de, self.chords_emb(np.expand_dims(pred, axis=-1))), axis=1)
            if return_str:
                # !! use pd + 1 because tk index starts from 1
                result.append([tk.index_word[pd + 1] for pd in pred])  # return chords string
            else:
                result.append(pred)  # return chords index
        result = np.transpose(np.array(result, dtype=object), (1, 0))  # (batch, out_seq_len)
        return result  # chords string

    def load_model(self, chords_emb_path, chords_extend_path, max_to_keep=5):
        self.cp_chords_emb = tf.train.Checkpoint(model=self.chords_emb, optimizer=self.optimizer)
        self.cp_manager_chords_emb = tf.train.CheckpointManager(
            self.cp_chords_emb, chords_emb_path, max_to_keep=max_to_keep)
        if self.cp_manager_chords_emb.latest_checkpoint:
            self.cp_chords_emb.restore(self.cp_manager_chords_emb.latest_checkpoint)
            print('Restored the latest chords_emb')

        self.cp_chords_extend = tf.train.Checkpoint(model=self.chords_extend, optimizer=self.optimizer)
        self.cp_manager_chords_extend = tf.train.CheckpointManager(
            self.cp_chords_extend, chords_extend_path, max_to_keep=max_to_keep)
        if self.cp_manager_chords_extend.latest_checkpoint:
            self.cp_chords_extend.restore(self.cp_manager_chords_extend.latest_checkpoint)
            print('Restored the latest chords_extender')

    def train_step(self, inputs, chords_emb=None):
        # x_in: numpy 3d array (batch, in_seq_len, 4):
        # x_tar_in: numpy 3d array (batch, out_seq_len, 4)
        # x_tar_out: numpy 3d array (batch, out_seq_len, 4)
        #   columns: chords_id (int), velocity, time_passed_since_last_note, chords_duration
        x_in, x_tar_in, x_tar_out = inputs

        with tf.GradientTape() as tape:
            mask_padding = None  # util.padding_mask(x_in[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_tar_in.shape[1])  # (out_seq_len, out_seq_len)

            #  x_in_nt: (batch, in_seq_len)
            #  x_tar_in_nt: (batch, out_seq_len)
            #  x_tar_out_nt: (batch, out_seq_len)
            x_in_nt, x_tar_in_nt, x_tar_out_nt = x_in[:, :, 0], x_tar_in[:, :, 0], x_tar_out[:, :, 0]
            #  x_in_nt: (batch, in_seq_len, embed_dim)
            #  x_tar_in_nt: (batch, out_seq_len, embed_dim)
            #  x_tar_out_nt: (batch, out_seq_len, embed_dim)
            x_in_nt, x_tar_in_nt, x_tar_out_nt = \
                    chords_emb(x_in_nt), chords_emb(x_tar_in_nt), chords_emb(x_tar_out_nt)

            # x_out_nt_pr: (batch, out_seq_len, out_chords_pool_size)
            x_out_nt_pr, _ = self.chords_extend((x_in_nt, x_tar_in_nt, mask_padding, mask_lookahead))
            loss_chords = util.loss_func_chords(x_tar_out[:, :, 0], x_out_nt_pr)
            variables_chords = self.chords_emb.trainable_variables + self.chords_extend.trainable_variables

            gradients = tape.gradient(loss_chords, variables_chords)
            self.optimizer.apply_gradients(zip(gradients, variables_chords))
            self.train_loss(loss_chords)
            return loss_chords

    def train(self, dataset, epochs=10, save_model_step=1, chords_emb_path=chords_emb_path,
              chords_extend_path=chords_extend_path, max_to_keep=5, print_batch=True, print_batch_step=10,
              print_epoch=True, print_epoch_step=5, warmup_steps=4000,
              optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
              chords_emb=None):

        learning_rate = util.CustomSchedule(self.embed_dim, warmup_steps)
        self.optimizer = optmzr(learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        # ---------------------- call back setting --------------------------
        self.load_model(chords_emb_path=chords_emb_path, chords_extend_path=chords_extend_path, max_to_keep=max_to_keep)

        # ---------------------- training --------------------------
        for epoch in range(epochs):
            self.train_loss.reset_states()
            start = time.time()
            for i, (x_in, x_tar_in, x_tar_out) in enumerate(dataset):
                losses = self.train_step((x_in, x_tar_in, x_tar_out), chords_emb)
                if print_batch:
                    if (i + 1) % print_batch_step == 0:
                        print('Epoch {} Batch {}: loss={:.4f}'.format(epoch + 1, i + 1, losses.numpy()))
            if print_epoch:
                if (epoch + 1) % print_epoch_step == 0:
                    print('Epoch {}: Loss = {:.4f}, Time used = {:.4f}'.format(
                        epoch + 1, self.train_loss.result(), time.time() - start))
            if (epoch + 1) % save_model_step == 0:
                self.cp_manager_chords_emb.save()
                print('Saved the latest chords_emb')
                self.cp_manager_chords_extend.save()
                print('Saved the latest chords_extend')


class TimeExtendPretrain(tf.keras.models.Model):

    def __init__(self, time_features=3, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3,
                 norm_epsilon=1e-6, transformer_dropout_rate=0.2):
        super(TimeExtendPretrain, self).__init__()
        self.time_features = time_features
        # 3 for [velocity, velocity, time since last start, chords duration]
        self.time_extend = transformer.Transformer(
            embed_dim=time_features, n_heads=1, out_chords_pool_size=time_features,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation,
            out_positive=True)

    def call(self, inputs, training=None, mask=None):
        # x_en: (batch_size, in_seq_len, time_features)
        # x_de: (batch_size, 1, time_features)
        x_en, out_seq_len = inputs
        batch_size = x_en.shape[0]
        x_de = tf.constant([[[0] * 3]] * batch_size, dtype=tf.float32)
        for i in range(out_seq_len):
            mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
            # x_out: (batch_size, 1, out_chords_pool_size)
            x_out, _ = self.time_extend((x_en, x_de, mask_padding, mask_lookahead))
            x_de = tf.concat((x_de, x_out[:, -1][:, tf.newaxis]), axis=1)
        return x_de[:, 1:, :]  # (batch_size, out_seq_len, time_features)

    def predict_time(self, x_en, out_seq_len, vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
                     return_denorm=True):
        # x_en: (batch_size, in_seq_len, 3)
        batch_size = x_en.shape[0]
        # init x_de: (batch_size, 1, 3)
        x_de = tf.constant([[[0] * 3]] * batch_size, dtype=tf.float32)
        result = []  # (out_seq_len, batch, 3)
        for i in range(out_seq_len):
            mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
            # x_out: (batch_size, 1, 3)
            x_out, _ = self.time_extend((x_en, x_de, mask_padding, mask_lookahead))
            pred = x_out[:, -1, :][:, tf.newaxis, :]  # only take the last prediction
            x_de = tf.concat((x_de, pred), axis=1)
            result.append(x_out[:, -1, :].numpy().tolist())  # only take the last prediction
        result = np.transpose(np.array(result, dtype=object), (1, 0, 2))  # (batch, out_seq_len, 3)
        if return_denorm:
            result = result * np.array([vel_norm, tmps_norm, dur_norm])
            result[:, 0] = np.clip(result[:, 0], 0, 127)  # squeeze velocity within limit
        return result

    def load_model(self, time_extend_path, max_to_keep=5):
        self.cp_time_extend = tf.train.Checkpoint(model=self.time_extend, optimizer=self.optimizer)
        self.cp_manager_time_extend = tf.train.CheckpointManager(
            self.cp_time_extend, time_extend_path, max_to_keep=max_to_keep)
        if self.cp_manager_time_extend.latest_checkpoint:
            self.cp_time_extend.restore(self.cp_manager_time_extend.latest_checkpoint)
            print('Restored the latest time_extend')

    def train_step(self, inputs):
        # x_in: numpy 3d array (batch, in_seq_len, 4):
        # x_tar_in: numpy 3d array (batch, out_seq_len, 4)
        # x_tar_out: numpy 3d array (batch, out_seq_len, 4)
        #   columns: chords_id (int), velocity, time_passed_since_last_note, chords_duration
        x_in, x_tar_in, x_tar_out = inputs
        
        with tf.GradientTape() as tape:
            mask_padding = None  # util.padding_mask(x_in[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_tar_in.shape[1])  # (out_seq_len, out_seq_len)
            #  x_in_tm: (batch, in_seq_len, 3)
            #  x_tar_in_tm: (batch, out_seq_len, 3)
            #  x_tar_out_tm: (batch, out_seq_len, 3)
            x_in_tm, x_tar_in_tm, x_tar_out_tm = x_in[:, :, 1:], x_tar_in[:, :, 1:], x_tar_out[:, :, 1:]
            # x_out_nt_pr: (batch, out_seq_len, 3)
            x_out_tm_pr, _ = self.time_extend((x_in_tm, x_tar_in_tm, mask_padding, mask_lookahead))
            # the ending value is 0, remove from loss calcultion
            loss_time = util.loss_func_time(x_tar_out_tm[:, :-1, :], x_out_tm_pr[:, :-1, :])
            variables_time = self.time_extend.trainable_variables

            gradients = tape.gradient(loss_time, variables_time)
            self.optimizer.apply_gradients(zip(gradients, variables_time))
            self.train_loss(loss_time)
            return loss_time

    def train(self, dataset, epochs=10, save_model_step=1, time_extend_path=time_extend_path, max_to_keep=5,
              print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=5, warmup_steps=4000,
              optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)):
        learning_rate = util.CustomSchedule(self.time_features, warmup_steps)
        self.optimizer = optmzr(learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        # ---------------------- call back setting --------------------------
        self.load_model(time_extend_path=time_extend_path, max_to_keep=max_to_keep)

        # ---------------------- training --------------------------
        for epoch in range(epochs):
            self.train_loss.reset_states()
            start = time.time()
            for i, (x_in, x_tar_in, x_tar_out) in enumerate(dataset):
                losses = self.train_step((x_in, x_tar_in, x_tar_out))
                if print_batch:
                    if (i + 1) % print_batch_step == 0:
                        print('Epoch {} Batch {}: Loss={:.4f}'.format(epoch+1, i+1, losses.numpy()))
            if print_epoch:
                if (epoch + 1) % print_epoch_step == 0:
                    print('Epoch {}: Loss = {:.4f}, Time used = {:.4f}'.format(
                        epoch + 1, self.train_loss.result(), time.time() - start))
            if (epoch + 1) % save_model_step == 0:
                self.cp_manager_time_extend.save()
                print('Saved the latest time_extend')

