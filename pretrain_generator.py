import util
import generator
import tensorflow as tf
import pickle as pkl
import os
import time
import numpy as np
import preprocess

tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

path_base = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/'
model_paths = {k: os.path.join(path_base, k) for k in
               ['chords_style', 'chords_syn', 'time_style', 'time_syn',
                'chords_disc', 'time_disc', 'comb_disc']}
result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'

tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_indexcer/chords_dict_mod.pkl'
tk = pkl.load(open(tk_path, 'rb'))


# -------------------------- pretrain chords embedder and projector --------------------------

#
# You can use the cosine distance between the embedding vectors in W and embedded_chars:
#
# # assume embedded_chars.shape == (batch_size, embedding_size)
# emb_distances = tf.matmul( # shape == (vocab_size, batch_size)
#     tf.nn.l2_normalize(W, dim=1),
#     tf.nn.l2_normalize(embedded_chars, dim=1),
#     transpose_b=True)
# token_ids = tf.argmax(emb_distances, axis=0) # shape == (batch_size)










in_seq_len = 16
out_seq_len = 1
true_data = util.load_true_data_pretrain_gen(
        tk, in_seq_len, out_seq_len, step=4, batch_size=200, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3,
        pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''])  # todo:


# -------------------------- pretrain chords syn --------------------------
in_dim = 512
embed_dim = 16
chstl_fc_layers = 4
chstl_activ = tf.keras.layers.LeakyReLU(alpha=0.1)
strt_dim = 5
chords_pool_size = 15000
n_heads = 4
init_knl = 3
max_pos = None
chsyn_fc_activation = tf.keras.layers.LeakyReLU(alpha=0.1)
chsyn_encoder_layers = 3
chsyn_decoder_layers = 3
chsyn_fc_layers = 3
chsyn_norm_epsilon = 1e-6
chsyn_embedding_dropout_rate = 0.2
chsyn_transformer_dropout_rate = 0.2


chords_syn = generator.ChordsSynthesis(
    out_chords_pool_size=chords_pool_size, embed_dim=embed_dim, init_knl=init_knl, strt_dim=strt_dim,
    n_heads=n_heads, max_pos=max_pos, fc_activation=chsyn_fc_activation, encoder_layers=chsyn_encoder_layers,
    decoder_layers=chsyn_decoder_layers, fc_layers=chsyn_fc_layers, norm_epsilon=chsyn_norm_epsilon,
    embedding_dropout_rate=chsyn_embedding_dropout_rate,
    transformer_dropout_rate=chsyn_transformer_dropout_rate)

# load existing model
warmup_steps = 400
learning_rate = util.CustomSchedule(embed_dim, warmup_steps)
optmzr = lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
max_to_keep = 5

ckpts_ch = tf.train.Checkpoint(model=chords_syn, optimizer=optmzr(learning_rate))
ckpt_managers_ch = tf.train.CheckpointManager(ckpts_ch, model_paths['chords_syn'], max_to_keep=max_to_keep)
if ckpt_managers_ch.latest_checkpoint:
    ckpts_ch.restore(ckpt_managers_ch.latest_checkpoint)
    print('restored')


# pretrain settingts
def pred_output_ch(x_in, out_seq_len, chords_syn, return_str=False):
    x_en = chords_syn.chords_emb(x_in[:, :, 0])  # (batch, strt_dim, embed_dim)
    result = []  # (out_seq_len, batch)
    pred_l = []
    x_de = tf.ones((x_en.shape[0], 1, embed_dim))  # (batch, 1, embed_dim)
    for i in range(out_seq_len):
        x_out_ch, _ = chords_syn.chords_extend(
            (x_en, x_de, None, None), noise_en=None, noise_de_1=None, noise_de_2=None)
        pred_l.append(x_out_ch)
        pred = tf.argmax(x_out_ch, -1)
        if return_str:
            result.append([tk.index_word[pd.numpy()[-1] + 1] for pd in pred])
        pred_ = chords_syn.chords_emb(pred[:, -1][:, tf.newaxis])
        x_en = tf.concat((x_en, pred_), axis=1)  # (batch, en_time_in+1, embed_dim)
    if return_str:
        return np.transpose(result, (1, 0)).astype(object)  # (batch, out_seq_len), string format
    # return (batch, out_seq_len, embed_dim), (batch, out_seq_len, chords_pool_size)
    return x_en[:, x_in.shape[1]:, :], tf.concat(pred_l, axis=1)


def pretrain_ch(true_data, optmzr, epochs, print_batch_step=10, print_epoch_step=1, save_model_step=1,
                save_nsamples=3):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    for epoch in range(epochs):
        train_loss.reset_states()
        start = time.time()
        for i, (x_in, x_tar) in enumerate(true_data):
            with tf.GradientTape() as tape:
                # (batch, out_seq_len, chords_pool_size)
                _, pred = pred_output_ch(x_in, out_seq_len, chords_syn, return_str=False)
                loss_chords = util.loss_func_chords(x_tar[:, :, 0], pred)
                variables_chords = chords_syn.trainable_variables
                gradients = tape.gradient(loss_chords, variables_chords)
                optmzr.apply_gradients(zip(gradients, variables_chords))
                train_loss(loss_chords)

            if (i + 1) % print_batch_step == 0:
                print('Epoch {} Batch {}: loss={:.4f}'.format(epoch + 1, i + 1, loss_chords.numpy()))
                # spl = tf.argmax(pred, -1).numpy()[np.random.choice(range(x_in.shape[0]), 3)]
                # chs = np.array([[tk.index_word[pd + 1] for pd in spl[i]] for i in range(len(spl))])
                # tms = np.multiply(np.array([[[1] * 3] * out_seq_len] * len(spl)),
                #                   np.array([vel_norm, tmps_norm, dur_norm]))
                # # ary: (save_nsamples, out_seq_len, 4)
                # ary = np.concatenate([chs[:, :, np.newaxis].astype(object), tms.astype(object)], axis=-1)
                # for j, ary_i in enumerate(ary):
                #     ary_i[:, 1] = np.clip(ary_i[:, 1], 0, 127)
                #     mid = preprocess.Conversion.arry2mid(ary_i)
                #     file_name = os.path.join(result_path, 'chords', 'ep{}_{}_{}.mid'.format(epoch + 1, i + 1, j))
                #     mid.save(file_name)
                # print('Saved {} fake samples'.format(save_nsamples))

        if (epoch + 1) % print_epoch_step == 0:
            print('Epoch {}: Loss = {:.4f}, Time used = {:.4f}'.format(
                epoch + 1, train_loss.result(), time.time() - start))
        if (epoch + 1) % save_model_step == 0:
            ckpt_managers_ch.save()
            print('Saved chords_syn')





        # ---------------------- call back setting --------------------------

# pretrain
pretrain_ch(true_data, optmzr(learning_rate), epochs=100, print_batch_step=1000, print_epoch_step=1, save_model_step=1,
            save_nsamples=1)


# -------------------------- pretrain time syn --------------------------






