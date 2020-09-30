from gan import WGAN
import tensorflow as tf
import os

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

path_base = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/'
model_paths = {k: os.path.join(path_base, k) for k in
               ['chords_embedder', 'chords_style', 'chords_syn', 'time_style', 'time_syn',
                'chords_disc', 'time_disc', 'comb_disc']}
result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'


# train on chords latent -------------------------------------------------
mode_ = 'chords'

out_seq_len = 4
gan_model = WGAN(in_dim=512, embed_dim=16, latent_std=1.0, strt_dim=3, n_heads=4, init_knl=3,
                 chstl_fc_layers=4, chstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1),
                 chsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                 chsyn_encoder_layers=3, chsyn_decoder_layers=3, chsyn_fc_layers=3, chsyn_norm_epsilon=1e-6,
                 chsyn_transformer_dropout_rate=0.2, chsyn_noise_std=0.1,
                 time_features=3, tmstl_fc_layers=4, tmstl_activ=tf.keras.layers.LeakyReLU(alpha=0.1),
                 tmsyn_encoder_layers=3, tmsyn_decoder_layers=3, tmsyn_fc_layers=3, tmsyn_norm_epsilon=1e-6,
                 tmsyn_transformer_dropout_rate=0.2, tmsyn_fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                 tmsyn_noise_std=0.1,
                 d_kernel_size=3, d_encoder_layers=1, d_decoder_layers=1, d_fc_layers=3, d_norm_epsilon=1e-6,
                 d_transformer_dropout_rate=0.2, d_fc_activation=tf.keras.activations.tanh,
                 d_out_dropout=0.3, d_recycle_fc_activ=tf.keras.activations.elu, mode_=mode_)
gan_model.load_model(model_paths, max_to_keep=5)
gan_model.set_trainable(train_chords_style=True, train_chords_syn=False,
                        train_time_style=False, train_time_syn=False,
                        train_disc=True)
gan_model.load_true_samples(step=out_seq_len, batch_size=50, out_seq_len=out_seq_len,
                            vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
                            pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''],
                            remove_same_chords=True)

optmzr = lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
gan_model.train(epochs=10, save_model_step=1, save_sample_step=1, print_batch=True, print_batch_step=100,
                print_epoch=True, print_epoch_step=5, disc_lr=0.0001, gen_lr=0.1,
                optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                result_path=result_path, out_seq_len=out_seq_len,
                save_nsamples=3, recycle_step=None)
# gan_model.gen_music(1)

# train on time latent -------------------------------------------------



