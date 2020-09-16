import os
import numpy as np
import util
import generator
import tensorflow as tf
import json
import pickle as pkl
import preprocess
tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

notes_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_embedder'
notes_extend_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_extender'
time_extend_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_extender'

tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_indexcer/notes_dict_final.pkl'
tk = pkl.load(open(tk_path, 'rb'))
notes_pool_size = len(json.loads(tk.get_config()['word_counts']))
print(notes_pool_size)


# pre-train notes extend -------------------------------------------------------------------------------------
in_seq_len, out_seq_len = 16, 64

dataset = util.load_true_data_pretrain_gen(
    tk, in_seq_len, out_seq_len, step=60, batch_size=100,
    vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
    pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['noc'])  # todo

notes_ext = generator.NotesExtend(
    out_notes_pool_size=15002, embed_dim=16, n_heads=4, max_pos=800, fc_activation="relu",
    encoder_layers=5, decoder_layers=5,
    fc_layers=3, norm_epsilon=1e-6, embedding_dropout_rate=0.2, transformer_dropout_rate=0.2)

notes_ext.train(dataset, epochs=1000, save_model_step=1,
                notes_emb_path=notes_emb_path, notes_extend_path=notes_extend_path,
                max_to_keep=5, print_batch=True, print_batch_step=100, print_epoch=True, print_epoch_step=1,
                warmup_steps=40000,
                optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9))


"""
notes_ext.optimizer = tf.keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
notes_ext.load_model(notes_emb_path, notes_extend_path, max_to_keep=5)

out_seq_len = 64
test_x_in, test_tar_in, test_tar_out = list(dataset.prefetch(1))[0]

spl = np.random.choice(range(test_x_in.shape[0]))

# pred
nts_in_i = test_x_in[spl, :, 0][np.newaxis, :]
nts = notes_ext.predict_notes(notes_ext.notes_emb(nts_in_i), tk, out_seq_len, return_str=True)

tms = np.array([[vel_norm, tmps_norm, dur_norm]] * out_seq_len)[np.newaxis, :, :]
ary = np.squeeze(np.concatenate([nts[:, :, np.newaxis].astype(object), 
    abs(tms).astype(object)], axis=-1), axis=0)  # (out_seq_len, 4)
ary = ary[(ary[:, 0] != '<start>') & (ary[:, 0] != '<end>')]
mid = preprocess.Conversion.arry2mid(ary)
mid.save('test_pred_notes.mid')

# true
nts_in_i_true = test_tar_out[spl, :, 0]
nts_true = np.array([tk.index_word[pd.numpy() + 1] for pd in nts_in_i_true])

ary = np.squeeze(np.concatenate([nts_true[:-1][np.newaxis, :, np.newaxis].astype(object), 
    abs(tms[:, :-1, :]).astype(object)], axis=-1), axis=0)  # (out_seq_len, 4)
ary = ary[(ary[:, 0] != '<start>') & (ary[:, 0] != '<end>')]
mid = preprocess.Conversion.arry2mid(ary)
mid.save('test_true_notes.mid')

# input 
nts_in_i = np.array([tk.index_word[pd.numpy() + 1] for pd in nts_in_i[0]])
ary = np.squeeze(np.concatenate([nts_in_i[np.newaxis, :, np.newaxis].astype(object), 
    abs(tms[:, :len(nts_in_i), :]).astype(object)], axis=-1), axis=0)  # (out_seq_len, 4)
ary = ary[(ary[:, 0] != '<start>') & (ary[:, 0] != '<end>')]
mid = preprocess.Conversion.arry2mid(ary)
mid.save('test_input_notes.mid')
"""


# pre-train time extender -------------------------------------------------------------------------------------
in_seq_len, out_seq_len = 16, 256  # 64

dataset = util.load_true_data_pretrain_gen(
    tk, in_seq_len, out_seq_len, step=30, batch_size=50,
    vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
    pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''])

time_ext = generator.TimeExtend(
    time_features=3, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3, norm_epsilon=1e-6,
    transformer_dropout_rate=0.2)

time_ext.train(dataset, epochs=10, save_model_step=1,
               time_extend_path=time_extend_path, max_to_keep=5,
               print_batch=True, print_batch_step=200, print_epoch=True, print_epoch_step=1, warmup_steps=4000,
               optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9))

