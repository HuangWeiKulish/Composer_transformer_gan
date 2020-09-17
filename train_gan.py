from gan import GAN
import pickle as pkl
import tensorflow as tf
import numpy as np

chords_latent_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_latent'
time_latent_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_latent'

chords_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_embedder'
chords_extend_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_extender'
time_extend_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_extender'

chords_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_discriminator'
time_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_discriminator'
combine_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/discriminator'

result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'

vel_norm = 64.0
tmps_norm = 0.12
dur_norm = 1.3

tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_indexcer/chords_dict_final.pkl'
tk = pkl.load(open(tk_path, 'rb'))


# train on time latent -------------------------------------------------
out_seq_len = 64  # 128
mode_ = 'time'
gan_model = GAN(strt_token_id=15001, out_chords_pool_size=15002, embed_dim=256, n_heads=4, max_pos=800,
                time_features=3, fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                g_encoder_layers=2, g_decoder_layers=2, g_fc_layers=3, g_norm_epsilon=1e-6,
                g_embedding_dropout_rate=0.2, g_transformer_dropout_rate=0.2,
                d_kernel_size=3, d_encoder_layers=2, d_decoder_layers=2, d_fc_layers=3, d_norm_epsilon=1e-6,
                d_transformer_dropout_rate=0.2,
                chords_latent_nlayers=4, chords_latent_dim_base=4, time_latent_nlayers=4, out_seq_len=out_seq_len,
                mode_=mode_)

gan_model.load_true_samples(tk, step=60, batch_size=50, vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
                            pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''])
epochs = 2
gan_model.train(epochs=epochs, save_model_step=1, save_sample_step=1,
                print_batch=True, print_batch_step=20, print_epoch=True, print_epoch_step=1,
                warmup_steps=500, disc_lr=0.00005, gen_lr=0.1,
                optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                chords_latent_path=chords_latent_path, time_latent_path=time_latent_path,
                chords_emb_path=chords_emb_path, chords_gen_path=chords_gen_path, time_gen_path=time_gen_path,
                chords_disc_path=chords_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
                result_path=result_path,
                train_ntlatent=False, train_tmlatent=True, train_ntemb=False,
                train_ntgen=False, train_tmgen=False, train_disc=True,
                save_chords_ltnt=False, save_time_ltnt=True, save_chords_emb=False,
                save_chords_gen=False, save_time_gen=False, save_disc=False, max_to_keep=5,
                load_disc=False,
                nt_ltnt_uniform=False, tm_ltnt_uniform=False,
                true_label_smooth=(0.95, 1.0), fake_label_smooth=(0.0, 0.15))
# Todo: switch load_disc=True and save_disc=True once generator loss is lowered



# train on chords latent -------------------------------------------------
out_seq_len = 16  # 128
mode_ = 'chords'
gan_model = GAN(strt_token_id=15001, out_chords_pool_size=15002, embed_dim=256, n_heads=4, max_pos=800,
                time_features=3, fc_activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                g_encoder_layers=2, g_decoder_layers=2, g_fc_layers=3, g_norm_epsilon=1e-6,
                g_embedding_dropout_rate=0.2, g_transformer_dropout_rate=0.2,
                d_kernel_size=3, d_encoder_layers=2, d_decoder_layers=2, d_fc_layers=3, d_norm_epsilon=1e-6,
                d_transformer_dropout_rate=0.2,
                chords_latent_nlayers=4, chords_latent_dim_base=4, time_latent_nlayers=4, out_seq_len=out_seq_len,
                mode_=mode_)
gan_model.load_true_samples(tk, step=16, batch_size=100, vel_norm=vel_norm, tmps_norm=tmps_norm, dur_norm=dur_norm,
                            pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['noc', 'sona'])  # Todo!!!!!
epochs = 20
gan_model.train(epochs=epochs, save_model_step=16, save_sample_step=1,
                print_batch=True, print_batch_step=20, print_epoch=True, print_epoch_step=1,
                warmup_steps=500, disc_lr=0.001, gen_lr=0.01,
                optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                chords_latent_path=chords_latent_path, time_latent_path=time_latent_path,
                chords_emb_path=chords_emb_path, chords_gen_path=chords_gen_path, time_gen_path=time_gen_path,
                chords_disc_path=chords_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
                result_path=result_path,
                train_ntlatent=True, train_tmlatent=False, train_ntemb=False,
                train_ntgen=False, train_tmgen=False, train_disc=True,
                save_chords_ltnt=True, save_time_ltnt=False, save_chords_emb=False,
                save_chords_gen=False, save_time_gen=False, save_disc=True, max_to_keep=10,
                load_disc=True,
                nt_ltnt_uniform=False, tm_ltnt_uniform=False,
                true_label_smooth=(0.8, 1.2), fake_label_smooth=(0.0, 0.1))  # fake_label_smooth=(0.0, 0.3)

# Todo: switch load_disc=True and save_disc=True once generator loss is lowered




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
