import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Reshape, Add, Dropout, Embedding, Lambda
import util
import transformer

tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

cp_embedder_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/transformer_gan/model/embedder'
cp_generator_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/transformer_gan/model/generator'


# todo: check values after embedding: has both postive and negative!!!!
class Embedder(tf.keras.Model):

    def __init__(self, notes_pool_size, max_pos, embed_dim, dropout_rate=0.2):
        super(Embedder, self).__init__()
        self.notes_pool_size = notes_pool_size
        self.max_pos = max_pos
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.lambda_0 = Lambda(lambda x_: x_[:, :, 0])
        self.lambda_1 = Lambda(lambda x_: x_[:, :, 1])
        self.embedder = Embedding(notes_pool_size, embed_dim)

    def call(self, x_in):
        # x_in dim: (batch, time_in, 2): 2 columns: notes, duration
        # --------------------------- split x into x_notes and x_duration ---------------------------
        x_notes = self.lambda_0(x_in)  # (batch, time_in), the string encoding of notes
        x_duration = self.lambda_1(x_in)  # (batch, time_in), the normalized duration
        # --------------------------- embedding ---------------------------
        notes_emb = self.embedder(x_notes)  # (batch, time_in, embed_dim)
        pos_emb = Embedder.positional_encoding(self.max_pos, self.embed_dim)  # (1, max_pos, embed_dim)
        # --------------------------- combine ---------------------------
        combine = notes_emb + pos_emb[:, : x_in.get_shape()[1], :] + x_duration[:, :, tf.newaxis]
        if self.dropout_rate is not None:
            combine = Dropout(self.dropout_rate)(combine)
        return combine  # (batch, time_in, embed_dim)

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
        self.embedder_de = Embedder(notes_pool_size=out_notes_pool_size, max_pos=de_max_pos, embed_dim=embed_dim,
                                    dropout_rate=embedding_dropout_rate)







    def call(self, x_en_melody, x_de_in, mask_padding, mask_lookahead):
        # x_en  # (batch, in_seq_len, embed_dim): x_en_in is the embed output of notes
        # x_de_in  # (batch, out_seq_len, 2): x_de_in is the un-embedded target

        # transformer for melody -----------------------------------------
        # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
        x_de_melody = self.embedder_de(x_de_in)  # (batch, out_seq_len, embed_dim)
        x_out_melody, all_weights_melody = transformer.Transformer.transformer(
            x_en=x_en_melody, x_de=x_de_melody, mask_padding=mask_padding, mask_lookahead=mask_lookahead,
            out_notes_pool_size=self.out_notes_pool_size, embed_dim=self.embed_dim,
            encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers, n_heads=self.n_heads,
            depth=self.depth, fc_layers=self.fc_layers, norm_epsilon=self.norm_epsilon,
            transformer_dropout_rate=self.transformer_dropout_rate, fc_activation=self.fc_activation, type='melody')

        # transformer for duration -----------------------------------------
        # x_out_duration: (batch, out_seq_len, 1)
        x_de_duration = tf.slice(x_de_in, [0, 0, 1], [tf.shape(x_de_in)[0], tf.shape(x_de_in)[1], 1])  # (batch, out_seq_len, 1)
        x_out_duration, all_weights_duration = transformer.Transformer.transformer(
            x_en=x_en_melody, x_de=x_de_duration, mask_padding=mask_padding, mask_lookahead=mask_lookahead,
            out_notes_pool_size=self.out_notes_pool_size, embed_dim=1,
            encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers, n_heads=1,
            depth=1, fc_layers=self.fc_layers, norm_epsilon=self.norm_epsilon,
            transformer_dropout_rate=self.transformer_dropout_rate, fc_activation=self.fc_activation, type='duration')
        x_out_duration = Dense(1, activation='relu')(x_out_duration)  # the last layer makes sure the output is positive

        return x_out_melody, all_weights_melody, x_out_duration, all_weights_duration


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
        self.learning_rate = util.CustomSchedule(embed_dim)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    def train_step(self, x_in, x_tar_in, x_tar_out, notes_dur_loss_weight=(1, 1)):
        # x_en_in: (batch, in_seq_len, 2)
        # x_de_in: (batch, out_seq_len, 2)
        with tf.GradientTape() as tape:
            x_en_ = self.embedder_en(x_in)  # (batch, in_seq_len, embed_dim)
            mask_padding = util.padding_mask(x_in[:, :, 0])  # (batch, 1, 1, seq_len)
            mask_lookahead = util.lookahead_mask(x_tar_in.shape[1])  # (seq_len, seq_len)

            # ------------------ predict ------------------------
            # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
            # x_out_duration: (batch, out_seq_len, 1)
            x_out_melody, all_weights_melody, x_out_duration, all_weights_duration = self.generator(
                x_en_, x_tar_in, mask_padding, mask_lookahead)
            # ------------------ calculate loss ------------------------
            loss_notes = util.loss_func_notes(x_tar_out[:, :, 0], x_out_melody)
            loss_duration = util.loss_func_duration(x_tar_out[:, :, 1], x_out_duration[:, :, 0])
            loss_combine = (loss_notes * notes_dur_loss_weight[0] +
                            loss_duration * notes_dur_loss_weight[1]) / sum(notes_dur_loss_weight)

        variables = self.embedder_en.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(loss_combine, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.train_loss(loss_combine)

        return loss_notes, loss_duration, loss_combine

    def train(self, epochs, dataset, notes_dur_loss_weight=(1, 1), save_model_step=10,
              cp_embedder_path=cp_embedder_path, cp_generator_path=cp_generator_path, max_cp_to_keep=5,
              print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=5):

        # ---------------------- call back setting --------------------------
        cp_embedder = tf.train.Checkpoint(model=self.embedder_en, optimizer=self.optimizer)
        cp_manager_embedder = tf.train.CheckpointManager(cp_embedder, cp_embedder_path, max_to_keep=max_cp_to_keep)
        if cp_manager_embedder.latest_checkpoint:
            cp_embedder.restore(cp_manager_embedder.latest_checkpoint)
            print('Restored the latest embedder')

        cp_generator = tf.train.Checkpoint(model=self.generator, optimizer=self.optimizer)
        cp_manager_generator = tf.train.CheckpointManager(cp_generator, cp_generator_path, max_to_keep=max_cp_to_keep)
        if cp_manager_generator.latest_checkpoint:
            cp_generator.restore(cp_manager_generator.latest_checkpoint)
            print('Restored the latest generator')

        # ---------------------- training --------------------------
        for epoch in range(epochs):
            self.train_loss.reset_states()
            start = time.time()
            for i, (x_in, x_tar_in, x_tar_out) in enumerate(dataset):
                loss_notes, loss_duration, loss_combine = self.train_step(
                    x_in, x_tar_in, x_tar_out, notes_dur_loss_weight)
                if print_batch:
                    if (i + 1) % print_batch_step == 0:
                        print('Epoch {} Batch {}: loss_notes={:.4f}, loss_duration={:.4f}, loss_combine={:.4f}'.format(
                            epoch+1, i+1, loss_notes.numpy(), loss_duration.numpy(), loss_combine.numpy()))
            if print_epoch:
                if (epoch + 1) % print_epoch_step == 0:
                    print('Epoch {}: Loss = {:.4f}, Time used = {:.4f}'.format(
                        epoch + 1, self.train_loss.result(), time.time() - start))
            if (epoch + 1) % save_model_step == 0:
                cp_manager_embedder.save()
                print('Saved the latest embedder')
                cp_manager_generator.save()
                print('Saved the latest generator')

    def predict(self, x_in, tk, out_seq_len, dur_denorm=20):
        # x_in: (batch, in_seq_len, 2), 2 columns: [notes in text format, duration in integer format]
        batch_size = x_in.shape[0]

        x_in_ = util.number_encode_text(x=x_in, tk=tk, dur_norm=dur_denorm).astype(np.float32)
        x_en_ = self.embedder_en(x_in_)  # (batch, in_seq_len, embed_dim)

        # init x_de_in
        x_de_in = tf.constant([[tk.word_index['<start>'], 0]] * batch_size, dtype=tf.float32)
        x_de_in = tf.expand_dims(x_de_in, 1)  # (batch, 1, 2)

        result = []  # (out_seq_len, batch, 2)
        for i in range(out_seq_len):
            mask_padding = util.padding_mask(x_in_[:, :, 0])
            mask_lookahead = util.lookahead_mask(x_de_in.shape[1])

            # x_out_melody: (batch, out_seq_len, out_notes_pool_size)
            # x_out_duration: (batch, out_seq_len, 1)
            x_out_melody, all_weights_melody, x_out_duration, all_weights_duration = self.generator(
                x_en_, x_de_in, mask_padding, mask_lookahead)

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

# Todo: change the output notes to 1 dim???




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