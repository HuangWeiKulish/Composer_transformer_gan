import util
import generator
import tensorflow as tf
import os
import time
import numpy as np
import gensim

tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

path_base = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/'
model_paths = {k: os.path.join(path_base, k) for k in
               ['chords_embedder', 'chords_style', 'chords_syn', 'time_style', 'time_syn',
                'chords_disc', 'time_disc', 'comb_disc']}
result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'

embedder = gensim.models.Word2Vec.load(os.path.join(model_paths['chords_embedder'], 'chords_embedder.model'))


# -------------------------- pretrain chords embedder and projector --------------------------
in_seq_len = 8
out_seq_len = 1
true_data = util.load_true_data_pretrain_gen(
        in_seq_len, out_seq_len, step=in_seq_len//2, batch_size=50,
        pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''])  # todo:


# -------------------------- pretrain chords syn --------------------------
embed_dim = 16
strt_dim = 5
n_heads = 4
init_knl = 3
chsyn_fc_activation = tf.keras.layers.LeakyReLU(alpha=0.1)
chsyn_encoder_layers = 3
chsyn_decoder_layers = 3
chsyn_fc_layers = 3
chsyn_norm_epsilon = 1e-6
chsyn_transformer_dropout_rate = 0.2
noise_std = 1.0

chords_syn = generator.ChordsSynthesis(
                embed_dim=embed_dim, init_knl=init_knl, strt_dim=strt_dim,
                n_heads=n_heads, fc_activation=chsyn_fc_activation, encoder_layers=chsyn_encoder_layers,
                decoder_layers=chsyn_decoder_layers, fc_layers=chsyn_fc_layers, norm_epsilon=chsyn_norm_epsilon,
                transformer_dropout_rate=chsyn_transformer_dropout_rate, noise_std=noise_std)

# load existing model
warmup_steps = 400
learning_rate = util.CustomSchedule(embed_dim, warmup_steps)
optmzr = lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
max_to_keep = 5


# negative quantity between -1 and 0,
# where 0 indicates orthogonality and values closer to -1 indicate greater similarity
loss_func = tf.keras.losses.cosine_similarity

# set check point
ckpts_ch = tf.train.Checkpoint(model=chords_syn, optimizer=optmzr(learning_rate))
ckpt_managers_ch = tf.train.CheckpointManager(ckpts_ch, model_paths['chords_syn'], max_to_keep=max_to_keep)
if ckpt_managers_ch.latest_checkpoint:
    ckpts_ch.restore(ckpt_managers_ch.latest_checkpoint)
    print('restored')


# functions
def ch2vec(ch_ary):
    # ch_ary: (batch, embed_dim)
    embeddings = [[embedder.wv[ch_i.numpy().decode('UTF-8')].tolist() for ch_i in ch_ary] for ch_ary in ch_ary]
    return embeddings


def pred_output_ch(x_in_emb, out_seq_len, chords_syn, return_str=False):
    # x_in_emb: (batch, seq_len, embed_dim)
    x_de = tf.ones((x_in_emb.shape[0], 1, embed_dim))  # (batch, 1, embed_dim)
    x_en = x_in_emb
    for i in range(out_seq_len):
        # x_out_ch: (batch, 1, embed_dim)
        x_out_ch, _ = chords_syn.chords_extend((x_in_emb, x_de, None, None), noise_en=None, noise_de=None)
        x_en = tf.concat((x_en, x_out_ch), axis=1)  # (batch, en_time_in+1, embed_dim)
    result = x_en[:, x_in_emb.shape[1]:, :]  # (batch, out_seq_len, embed_dim)
    if return_str:
        return np.array([[embedder.wv.similar_by_vector(ch, topn=1)[0][0] for ch in ch_p] for ch_p in result.numpy()])
    return result


def pretrain_ch(true_data, optmzr, epochs, print_batch_step=10, print_epoch_step=1, save_model_step=1):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    for epoch in range(epochs):
        train_loss.reset_states()
        start = time.time()
        for i, (x_in, x_tar) in enumerate(true_data):
            x_in_emb = tf.constant(ch2vec(x_in[:, :, 0]))
            x_tar_emb = tf.constant(ch2vec(x_tar[:, :, 0]))
            with tf.GradientTape() as tape:
                # (batch, out_seq_len, embed_dim)
                pred = pred_output_ch(x_in_emb, out_seq_len, chords_syn, return_str=False)
                loss_chords = loss_func(x_tar_emb, pred)
                variables_chords = chords_syn.trainable_variables
                gradients = tape.gradient(loss_chords, variables_chords)
                optmzr.apply_gradients(zip(gradients, variables_chords))
                train_loss(loss_chords)
            if (i + 1) % print_batch_step == 0:
                print('Epoch {} Batch {}: loss={:.4f}'.format(epoch + 1, i + 1, loss_chords.numpy().mean()))
        if (epoch + 1) % print_epoch_step == 0:
            print('Epoch {}: Loss = {:.4f}, Time used = {:.4f}'.format(
                epoch + 1, train_loss.result(), time.time() - start))
        if (epoch + 1) % save_model_step == 0:
            ckpt_managers_ch.save()
            print('Saved chords_syn')


# pretrain
pretrain_ch(true_data, optmzr(learning_rate), epochs=100, print_batch_step=5000, print_epoch_step=1, save_model_step=1)


# -------------------------- pretrain time syn --------------------------






