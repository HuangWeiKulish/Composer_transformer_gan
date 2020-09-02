import numpy as np
import time
import tensorflow as tf
import util
import transformer

tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

cp_embedder_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/transformer_gan/model/embedder'
cp_generator_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/transformer_gan/model/generator'


class NotesEmbedder(tf.keras.Model):

    def __init__(self, notes_pool_size, max_pos, embed_dim=256, dropout_rate=0.2):
        super(NotesEmbedder, self).__init__()
        self.notes_pool_size = notes_pool_size
        self.max_pos = max_pos
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.embedder = tf.keras.layers.Embedding(notes_pool_size, embed_dim)

    def call(self, x_in):
        # x_in dim: (batch, time_in): notes index
        # --------------------------- split x into x_notes and x_duration ---------------------------
        #x_notes = tf.keras.layers.Lambda(lambda x_: x_[:, :, 0])(x_in)  # (batch, time_in), the string encoding of notes
        # --------------------------- embedding ---------------------------
        notes_emb = self.embedder(x_in)  # (batch, time_in, embed_dim)
        pos_emb = NotesEmbedder.positional_encoding(self.max_pos, self.embed_dim)  # (1, max_pos, embed_dim)
        # --------------------------- combine ---------------------------
        combine = notes_emb + pos_emb[:, : x_in.get_shape()[1], :]  # (batch, time_in, embed_dim)
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
        angle_rads = NotesEmbedder.get_angles(
            np.arange(max_pos)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)


class NotesLatent(tf.keras.Model):

    def __init__(self, nlayers=4, dim_base=4):
        # generate latent vector of dimension; (batch, in_seq_len, embed_dim)
        # example: input random vector (batch, 16) ==> output latent vector (batch, 16, 256)
        super(NotesLatent, self).__init__()
        self.fcs = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(dim_base**i, use_bias=True) for i in range(1, nlayers+1)])

    def call(self, x_in):
        # x_in: (batch, in_seq_len, 1)
        return self.fcs(x_in)


class OtherLatent(tf.keras.Model):

    def __init__(self, nlayers=4):
        # generate latent vector of dimension; (batch, in_seq_len, in_features)
        # example: input random vector (batch, 16) ==> output latent vector (batch, 16, 256)
        super(OtherLatent, self).__init__()
        self.fcs = tf.keras.models.Sequential(
            [tf.keras.layers.Conv1D(filters=4, kernel_size=3, padding='same') for i in range(1, nlayers+1)])

    def call(self, x_in):
        # x_in: (batch, in_seq_len, 1)
        return self.fcs(x_in)


class Generator(tf.keras.Model):

    def __init__(self, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3, norm_epsilon=1e-6,
                 transformer_dropout_rate=0.2, mode_='notes'):
        super(Generator, self).__init__()
        #assert embed_dim % n_heads == 0
        if mode_ == 'notes':
            embed_dim = 256
            n_heads = 4
            out_notes_pool_size = 15002
        else:
            # 3 for [velocity, velocity, time since last start, notes duration]
            embed_dim = 3
            n_heads = 1
            out_notes_pool_size = 3
        self.gen = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=out_notes_pool_size,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)

    def call(self, x_en, x_de, mask_padding, mask_lookahead):
        # if mode_ == 'notes'
        #   x_en: (batch, in_seq_len, embed_dim)
        #   x_de: (batch, out_seq_len, embed_dim)
        #   x_out: (batch, out_seq_len, out_notes_pool_size)
        # else:
        #   x_en: (batch, in_seq_len, 3)
        #   x_de: (batch, out_seq_len, 3)
        #   x_out: (batch, out_seq_len, 3)
        x_out, all_weights = self.gen(x_en=x_en, x_de=x_de, mask_padding=mask_padding, mask_lookahead=mask_lookahead)
        return x_out, all_weights

    def predict_notes(self, x_en, tk, emb_model, out_seq_len, return_str=True):
        # emb_model is the embedding model
        # x_en: (batch_size, in_seq_len, embed_dim)
        # x_de: batch_size, 1, embed_dim)
        batch_size = x_en.shape[0]
        x_de = tf.constant([[tk.word_index['<start>']]] * batch_size, dtype=tf.float32)
        x_de = emb_model(x_de)
        result = []  # (out_seq_len, batch)
        for i in range(out_seq_len):
            mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
            # x_out: (batch_size, 1, out_notes_pool_size)
            x_out, _ = self.gen(x_en=x_en, x_de=x_de, mask_padding=mask_padding, mask_lookahead=mask_lookahead)
            # translate prediction to text
            pred = tf.argmax(x_out, -1)
            pred = [nid for nid in pred.numpy()[:, -1]]  # len = batch, take the last prediction
            # append pred to x_de:
            x_de = tf.concat((x_de, emb_model(np.expand_dims(pred, axis=-1))), axis=1)
            if return_str:
                result.append([tk.index_word[pd] for pd in pred])  # return notes string
            else:
                result.append(pred)  # return notes index
        result = np.transpose(np.array(result, dtype=object), (1, 0))  # (batch, out_seq_len)
        return result  # notes string

    def predict_other(self, x_en, out_seq_len, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3, return_denorm=True):
        # x_en: (batch_size, in_seq_len, 3)
        batch_size = x_en.shape[0]
        # init x_de: (batch_size, 1, 3)
        x_de = tf.constant([[[0] * 3]] * batch_size, dtype=tf.float32)
        result = []  # (out_seq_len, batch, 3)
        for i in range(out_seq_len):
            mask_padding = None  # util.padding_mask(x_en[:, :, 0])  # (batch, 1, 1, in_seq_len)
            mask_lookahead = util.lookahead_mask(x_de.shape[1])  # (len(x_de_in), len(x_de_in))
            # x_out: (batch_size, 1, 3)
            x_out, _ = self.gen(x_en=x_en, x_de=x_de, mask_padding=mask_padding, mask_lookahead=mask_lookahead)
            pred = x_out[:, -1, :][:, tf.newaxis, :]  # only take the last prediction
            x_de = tf.concat((x_de, pred), axis=1)
            result.append(x_out[:, -1, :].numpy().tolist())  # only take the last prediction
        result = np.transpose(np.array(result, dtype=object), (1, 0, 2))  # (batch, out_seq_len, 3)
        if return_denorm:
            result = result * np.array([vel_norm, tmps_norm, dur_norm])
        return result


"""
x_en = x_in[:, :, 0]
x_de = x_tar_in[:, :, 0]

emb_model = NotesEmbedder(out_notes_pool_size, max_pos=800, embed_dim=256, dropout_rate=0.2)
x_en = emb_model(x_en)
x_en.shape

x_de = emb_model(x_de)
x_de.shape

# x_in, x_tar_in, x_tar_out
x_en = x_in[:, :, 1:]
x_en.shape
"""

#
# class GeneratorPretrain(tf.keras.Model):
#
#     def __init__(self, en_max_pos=5000, de_max_pos=10000, embed_dim=256, n_heads=4, in_notes_pool_size=150000,
#                  out_notes_pool_size=150000, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3,
#                  norm_epsilon=1e-6, transformer_dropout_rate=0.2, embedding_dropout_rate=0.2, beta_1=0.9, beta_2=0.98,
#                  epsilon=1e-9):
#         super(GeneratorPretrain, self).__init__()
#         self.embedder_en = Embedder(
#             notes_pool_size=in_notes_pool_size, max_pos=en_max_pos, embed_dim=embed_dim,
#             dropout_rate=embedding_dropout_rate)
#         self.generator = Generator(
#             de_max_pos=de_max_pos, embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=out_notes_pool_size,
#             fc_activation=fc_activation, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
#             fc_layers=fc_layers, norm_epsilon=norm_epsilon, transformer_dropout_rate=transformer_dropout_rate,
#             embedding_dropout_rate=embedding_dropout_rate)
#
#         self.train_loss = tf.keras.metrics.Mean(name='train_loss')
#         self.learning_rate = util.CustomSchedule(embed_dim)
#         self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
#
#     def train_step(self, x_in, x_tar_in, x_tar_out, notes_dur_loss_weight=(1, 1)):
#         # x_in: numpy 3d array (batch, in_seq_len, 2): column 1: notes id (converted to int), column 2: normalised duration
#         # x_tar_in: numpy 3d array (batch, out_seq_len, 2)
#         # x_tar_out: numpy 3d array (batch, x_tar_out, 2)
#         with tf.GradientTape() as tape:
#             x_en_melody = self.embedder_en(x_in)  # (batch, in_seq_len, embed_dim)
#             x_en_duration = x_in[:, :, 1][:, :, tf.newaxis]  # get the duration (batch, in_seq_len)
#             mask_padding = util.padding_mask(x_in[:, :, 0])  # (batch, 1, 1, seq_len)
#             mask_lookahead = util.lookahead_mask(x_tar_in.shape[1])  # (seq_len, seq_len)
#
#             # ------------------ predict ------------------------
#             # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
#             # x_out_duration: (batch, out_seq_len, 1)
#             x_out_melody, all_weights_melody, x_out_duration, all_weights_duration = self.generator(
#                 x_en_melody, x_en_duration, x_tar_in, mask_padding, mask_lookahead)
#
#             # ------------------ calculate loss ------------------------
#             loss_notes = util.loss_func_notes(x_tar_out[:, :, 0], x_out_melody)
#             loss_duration = util.loss_func_duration(x_tar_out[:, :, 1], x_out_duration[:, :, 0])
#             loss_combine = (loss_notes * notes_dur_loss_weight[0] +
#                             loss_duration * notes_dur_loss_weight[1]) / sum(notes_dur_loss_weight)
#
#         variables = self.embedder_en.trainable_variables + self.generator.trainable_variables
#         gradients = tape.gradient(loss_combine, variables)
#         self.optimizer.apply_gradients(zip(gradients, variables))
#         self.train_loss(loss_combine)
#
#         return loss_notes, loss_duration, loss_combine
#
#     def train(self, epochs, dataset, notes_dur_loss_weight=(1, 1), save_model_step=10,
#               cp_embedder_path=cp_embedder_path, cp_generator_path=cp_generator_path, max_cp_to_keep=5,
#               print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=5):
#
#         # ---------------------- call back setting --------------------------
#         cp_embedder = tf.train.Checkpoint(model=self.embedder_en, optimizer=self.optimizer)
#         cp_manager_embedder = tf.train.CheckpointManager(cp_embedder, cp_embedder_path, max_to_keep=max_cp_to_keep)
#         if cp_manager_embedder.latest_checkpoint:
#             cp_embedder.restore(cp_manager_embedder.latest_checkpoint)
#             print('Restored the latest embedder')
#
#         cp_generator = tf.train.Checkpoint(model=self.generator, optimizer=self.optimizer)
#         cp_manager_generator = tf.train.CheckpointManager(cp_generator, cp_generator_path, max_to_keep=max_cp_to_keep)
#         if cp_manager_generator.latest_checkpoint:
#             cp_generator.restore(cp_manager_generator.latest_checkpoint)
#             print('Restored the latest generator')
#
#         # ---------------------- training --------------------------
#         for epoch in range(epochs):
#             self.train_loss.reset_states()
#             start = time.time()
#             for i, (x_in, x_tar_in, x_tar_out) in enumerate(dataset):
#                 loss_notes, loss_duration, loss_combine = self.train_step(
#                     x_in, x_tar_in, x_tar_out, notes_dur_loss_weight)
#                 if print_batch:
#                     if (i + 1) % print_batch_step == 0:
#                         print('Epoch {} Batch {}: loss_notes={:.4f}, loss_duration={:.4f}, loss_combine={:.4f}'.format(
#                             epoch+1, i+1, loss_notes.numpy(), loss_duration.numpy(), loss_combine.numpy()))
#             if print_epoch:
#                 if (epoch + 1) % print_epoch_step == 0:
#                     print('Epoch {}: Loss = {:.4f}, Time used = {:.4f}'.format(
#                         epoch + 1, self.train_loss.result(), time.time() - start))
#             if (epoch + 1) % save_model_step == 0:
#                 cp_manager_embedder.save()
#                 print('Saved the latest embedder')
#                 cp_manager_generator.save()
#                 print('Saved the latest generator')
#
#     def predict(self, x_in, tk, out_seq_len, dur_denorm=20):
#         # x_in: (batch, in_seq_len, 2), 2 columns: [notes (string), duration (int)]
#         x_in_ = util.number_encode_text(x=x_in, tk=tk, dur_norm=dur_denorm).astype(np.float32)  # string to int id
#         x_en_melody = self.embedder_en(x_in_)  # (batch, in_seq_len, embed_dim)
#         x_en_duration = x_in_[:, :, 1][:, :, tf.newaxis]  # get the duration (batch, in_seq_len, 1)
#
#         # result: (batch, out_seq_len, 2); 2 columns: [notes1 (string), dur1 (int)]
#         result = self.generator.predict(x_en_melody=x_en_melody, x_en_duration=x_en_duration, tk=tk,
#                                         out_seq_len=out_seq_len, dur_denorm=dur_denorm)
#         return result


"""
de_max_pos=8000
embed_dim=256
n_heads=4
out_notes_pool_size=15002
fc_activation="relu"
encoder_layers=2
decoder_layers=2
fc_layers=3
norm_epsilon=1e-6
transformer_dropout_rate=0.2
embedding_dropout_rate=0.2

gen = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=out_notes_pool_size,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)
"""

"""
de_max_pos=8000
embed_dim=3
n_heads=1
out_notes_pool_size=3
fc_activation="relu"
encoder_layers=2
decoder_layers=2
fc_layers=3
norm_epsilon=1e-6
transformer_dropout_rate=0.2
embedding_dropout_rate=0.2

gen = transformer.Transformer(
            embed_dim=embed_dim, n_heads=n_heads, out_notes_pool_size=out_notes_pool_size,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers, fc_layers=fc_layers,
            norm_epsilon=norm_epsilon, dropout_rate=transformer_dropout_rate, fc_activation=fc_activation)
"""