from gan import GAN
import pickle as pkl
import tensorflow as tf
import numpy as np
import os

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

# random sample result
result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'

tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_indexcer/chords_dict_gan.pkl'
tk = pkl.load(open(tk_path, 'rb'))

embed_dim = 16
time_features = 3
in_dim = 512
n_heads = 4
strt_dim = 4
strt_token_id = 15001
chords_pool_size = 15002
chords_max_pos = 800
chstl_fc_layers = 4
chstl_activ = tf.keras.layers.LeakyReLU(alpha=0.1)
tmstl_fc_layers = 4
tmstl_activ = tf.keras.layers.LeakyReLU(alpha=0.1)
embedding_dropout_rate = 0.2
chsyn_kernel_size = 3
chsyn_fc_activation = tf.keras.layers.LeakyReLU(alpha=0.1)
chsyn_encoder_layers = 3
chsyn_decoder_layers = 3
chsyn_fc_layers = 3
chsyn_norm_epsilon = 1e-6
chsyn_transformer_dropout_rate = 0.2
chsyn_activ = tf.keras.layers.LeakyReLU(alpha=0.1)
tmsyn_kernel_size = 3
tmsyn_fc_activation = tf.keras.layers.LeakyReLU(alpha=0.1)
tmsyn_encoder_layers = 3
tmsyn_decoder_layers = 3
tmsyn_fc_layers = 3
tmsyn_norm_epsilon = 1e-6
tmsyn_transformer_dropout_rate = 0.2
tmsyn_activ = tf.keras.layers.LeakyReLU(alpha=0.1)
d_kernel_size = 3
d_encoder_layers = 2
d_decoder_layers = 2
d_fc_layers = 3
d_norm_epsilon = 1e-6
d_transformer_dropout_rate = 0.2
d_fc_activation = tf.keras.layers.LeakyReLU(alpha=0.1)
d_out_dropout = 0.3

# train on time latent -------------------------------------------------
mode_ = 'chords'
gan_model = GAN(out_seq_len_list=out_seq_len_list, embed_dim=embed_dim, time_features=time_features, in_dim=in_dim,
                n_heads=n_heads, strt_dim=strt_dim, strt_token_id=strt_token_id, chords_pool_size=chords_pool_size,
                chords_max_pos=chords_max_pos, mode_=mode_, chstl_fc_layers=chstl_fc_layers, chstl_activ=chstl_activ,
                tmstl_fc_layers=tmstl_fc_layers, tmstl_activ=tmstl_activ, embedding_dropout_rate=embedding_dropout_rate,
                chsyn_kernel_size=chsyn_kernel_size, chsyn_fc_activation=chsyn_fc_activation,
                chsyn_encoder_layers=chsyn_encoder_layers, chsyn_decoder_layers=chsyn_decoder_layers,
                chsyn_fc_layers=chsyn_fc_layers, chsyn_norm_epsilon=chsyn_norm_epsilon,
                chsyn_transformer_dropout_rate=chsyn_transformer_dropout_rate, chsyn_activ=chsyn_activ,
                tmsyn_kernel_size=tmsyn_kernel_size, tmsyn_fc_activation=tmsyn_fc_activation,
                tmsyn_encoder_layers=tmsyn_encoder_layers, tmsyn_decoder_layers=tmsyn_decoder_layers,
                tmsyn_fc_layers=tmsyn_fc_layers, tmsyn_norm_epsilon=tmsyn_norm_epsilon,
                tmsyn_transformer_dropout_rate=tmsyn_transformer_dropout_rate, tmsyn_activ=tmsyn_activ,
                d_kernel_size=d_kernel_size, d_encoder_layers=d_encoder_layers, d_decoder_layers=d_decoder_layers,
                d_fc_layers=d_fc_layers, d_norm_epsilon=d_norm_epsilon,
                d_transformer_dropout_rate=d_transformer_dropout_rate, d_fc_activation=d_fc_activation,
                d_out_dropout=d_out_dropout)

gan_model.load_true_samples(tk, step=out_seq_len_list[-1], batch_size=50,
                            vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
                            pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''])  # todo!!

gan_model.load_model(const_path=const_path, chords_emb_path=chords_emb_path, ini_layer_path=ini_layer_path,
                     style_paths=style_paths, chords_syn_paths=chords_syn_paths, time_syn_paths=time_syn_paths,
                     disc_paths=disc_paths, max_to_keep=5)

gan_model.set_trainable(train_emb=True, train_chords_ini=True, train_time_ini=True,
                        train_style_ch=True, train_style_tm=True, train_syn_ch={k: True for k in out_seq_len_list},
                        train_syn_tm={k: True for k in out_seq_len_list}, train_disc=True)


gan_model.train(tk, epochs=10, save_model_step=1, save_sample_step=1, print_batch=True, print_batch_step=50,
                print_epoch=True, print_epoch_step=5, disc_lr=0.0001, gen_lr=0.2,
                optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                result_path=result_path, save_nsamples=3,
                true_label_smooth=(0.9, 1.0), fake_label_smooth=(0.0, 0.1), recycle_step=3)  # todo!!


# import matplotlib.pyplot as plt
# data = list(gan_model.true_data.prefetch(100))[-1].numpy()[:, 0]
# data = gan_model.chords_emb(data)
# plt.hist(data.numpy().flatten(), bins=100)
# plt.show()



"""

import matplotlib.pyplot as plt
# nt_ltnt = np.random.uniform(-1.5, 1.5, (10, 16, 16))
import util
nt_ltnt = util.latant_vector(10, 16, 16, mean_=1.0, std_=0.5)
plt.hist(nt_ltnt.flatten(), bins=100)
plt.title('from normal chords latent')
plt.show()

vals = gan_model.chords_latent(np.random.uniform(0, 1.5, (100, 16, 16))).numpy()
plt.hist(vals.flatten(), bins=100)
plt.title('from uniform chords latent')
plt.show()

dt = [list(gan_model.true_data.prefetch(1))[0][:, :, 0].numpy() for i in range(100)]
vals = gan_model.chords_gen.chords_emb(np.concatenate(dt)).numpy()
plt.hist(vals.flatten(), bins=100)
plt.title('from real music embedding')
plt.show()   # multiple peaks !!
"""
