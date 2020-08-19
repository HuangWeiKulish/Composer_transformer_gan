import os
import numpy as np
import util
import generator
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

cp_embedder_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/transformer_gan/model/embedder'
cp_generator_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/transformer_gan/model/generator'

tk = tf.keras.preprocessing.text.Tokenizer(filters='')
tk, dataset = util.load_true_data(tk, in_seq_len=32, out_seq_len=128, step=60, batch_size=20, dur_denorm=20,
                   filepath_list=['/Users/Wei/Desktop/piano_classic/Chopin_array'], name_substr_list=['sonat'])

generator = generator.GeneratorPretrain(
    en_max_pos=5000, de_max_pos=10000, embed_dim=256, n_heads=4, in_notes_pool_size=150000, out_notes_pool_size=150000,
    fc_activation="relu", encoder_layers=2, decoder_layers=2, fc_layers=3, norm_epsilon=1e-6,
    transformer_dropout_rate=0.2, embedding_dropout_rate=0.2, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

generator.train(epochs=1, dataset=dataset, notes_dur_loss_weight=(1, 1), save_model_step=1,
                cp_embedder_path=cp_embedder_path, cp_generator_path=cp_generator_path, max_cp_to_keep=5,
                print_batch=True, print_batch_step=10, print_epoch=True, print_epoch_step=1)

# Todo: save tk!!!
# Todo: summarize and get unique notes!!!!

"""
dur_denorm=20
x, _, _ = list(dataset.prefetch(1).as_numpy_iterator())[0]
notes_dur = [[util.inds2notes(tk, nid, default='p'), dur * dur_denorm] for nid, dur in
             zip(x[0, :, 0], x[0, :, 1])]  # [[notes1, dur1], ...] for all batches 
notes_dur = np.expand_dims(np.array(notes_dur, dtype=object), axis=0)
generator.predict(x_in=notes_dur, tk=tk, out_seq_len=128, dur_denorm=20)

# -----------------
tmp = util.number_encode_text(notes_dur, tk, dur_norm=20)
tmp = generator.embedder_en(tmp)

import matplotlib.pyplot as plt
plt.hist(tmp.numpy().flatten(), bins=50)
"""






