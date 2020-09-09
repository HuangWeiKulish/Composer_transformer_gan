import os
import numpy as np
import util
import generator
import tensorflow as tf
import json
import pickle as pkl
tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

notes_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_embedder'
notes_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_generator'
time_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_generator'

tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_indexcer/notes_dict_final.pkl'
tk = pkl.load(open(tk_path, 'rb'))
notes_pool_size = len(json.loads(tk.get_config()['word_counts']))
print(notes_pool_size)


# pre-train notes generator -------------------------------------------------------------------------------------
in_seq_len, out_seq_len = 16, 128  # 64

dataset = util.load_true_data_pretrain_gen(
    tk, in_seq_len, out_seq_len, step=60, batch_size=50, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3,
    pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''])

notes_gen = generator.NotesGenerator(
    out_notes_pool_size=15002, embed_dim=256, n_heads=4, max_pos=800, fc_activation="relu", encoder_layers=2,
    decoder_layers=2, fc_layers=3, norm_epsilon=1e-6, embedding_dropout_rate=0.2, transformer_dropout_rate=0.2)

notes_gen.train(dataset, epochs=2, save_model_step=1, notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path,
                max_to_keep=5, print_batch=True, print_batch_step=1, print_epoch=True, print_epoch_step=5,
                warmup_steps=4000,
                optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9))

# pre-train time generator -------------------------------------------------------------------------------------
in_seq_len, out_seq_len = 16, 256  # 64

dataset = util.load_true_data_pretrain_gen(
    tk, in_seq_len, out_seq_len, step=60, batch_size=50, vel_norm=64.0, tmps_norm=0.12,
    dur_norm=1.3, pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''])

time_gen = generator.TimeGenerator(
    time_features=3, fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3, norm_epsilon=1e-6,
    transformer_dropout_rate=0.2)

time_gen.train(dataset, epochs=10, save_model_step=1, time_gen_path=time_gen_path, max_to_keep=5,
              print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=5, warmup_steps=4000,
              optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9))


"""
x_in, x_tar_in, x_tar_out = list(dataset.prefetch(1).as_numpy_iterator())[0]
pred_tm = time_gen.predict_time(x_in[:, :, 1:], 64, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3, return_denorm=True)

vel_norm=64.0; tmps_norm=0.12; dur_norm=1.3
true_tm = x_tar_out[:, :, 1:] * np.array([vel_norm, tmps_norm, dur_norm])

print(pred_tm.mean(axis=(0, 1)), true_tm.mean(axis=(0, 1)))
print(pred_tm.min(axis=(0, 1)), true_tm.min(axis=(0, 1)))
print(pred_tm.max(axis=(0, 1)), true_tm.max(axis=(0, 1)))
"""

"""
x_in, x_tar_in, x_tar_out = list(dataset.prefetch(1).as_numpy_iterator())[0]

out_notes_pool_size=15002
embed_dim=256
n_heads=4
max_pos=800
time_features=3
fc_activation="relu"
encoder_layers=2
decoder_layers=2
fc_layers=3
norm_epsilon=1e-6
embedding_dropout_rate=0.2
transformer_dropout_rate=0.2
mode_='notes'

epochs=5
nt_tm_loss_weight=(1, 1)
save_model_step=10
max_to_keep=5
print_batch=True
print_batch_step=1
print_epoch=True
print_epoch_step=1
lr_tm=0.01
warmup_steps=4000
optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
"""
