import os
import numpy as np
import util
import generator
import tensorflow as tf
import json
import pickle as pkl
tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

notes_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_embedder'
notes_extend_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_extend'
time_extend_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_extender'

tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_indexcer/notes_dict_final.pkl'
tk = pkl.load(open(tk_path, 'rb'))
notes_pool_size = len(json.loads(tk.get_config()['word_counts']))
print(notes_pool_size)


# pre-train notes extend -------------------------------------------------------------------------------------
in_seq_len, out_seq_len = 16, 64  # 64

dataset = util.load_true_data_pretrain_gen(
    tk, in_seq_len, out_seq_len, step=20, batch_size=50,
    vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
    pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''])

notes_ext = generator.NotesExtend(
    out_notes_pool_size=15002, embed_dim=16, n_heads=4, max_pos=800, fc_activation="relu", encoder_layers=2,
    decoder_layers=2, fc_layers=3, norm_epsilon=1e-6, embedding_dropout_rate=0.2, transformer_dropout_rate=0.2)

notes_ext.train(dataset, epochs=40, save_model_step=1,
                notes_emb_path=notes_emb_path, notes_extend_path=notes_extend_path,
                max_to_keep=5, print_batch=True, print_batch_step=100, print_epoch=True, print_epoch_step=1,
                warmup_steps=4000,
                optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9))


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

