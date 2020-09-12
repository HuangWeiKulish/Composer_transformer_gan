from gan import GAN
import pickle as pkl
import tensorflow as tf

notes_latent_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_latent'
time_latent_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_latent'

notes_emb_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_embedder'
notes_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_generator'
time_gen_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_generator'

notes_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_discriminator'
time_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/time_discriminator'
combine_disc_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/discriminator'

result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/result'

tk_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_indexcer/notes_dict_final.pkl'
tk = pkl.load(open(tk_path, 'rb'))


# train on time latent -------------------------------------------------
out_seq_len = 64  # 128
mode_ = 'time'
gan_model = GAN(strt_token_id=15001, out_notes_pool_size=15002, embed_dim=256, n_heads=4, max_pos=800,
                time_features=3, fc_activation="relu",
                g_encoder_layers=2, g_decoder_layers=2, g_fc_layers=3, g_norm_epsilon=1e-6,
                g_embedding_dropout_rate=0.2, g_transformer_dropout_rate=0.2,
                d_kernel_size=3, d_encoder_layers=2, d_decoder_layers=2, d_fc_layers=3, d_norm_epsilon=1e-6,
                d_transformer_dropout_rate=0.2,
                notes_latent_nlayers=4, notes_latent_dim_base=4, time_latent_nlayers=4, out_seq_len=out_seq_len,
                mode_=mode_)

gan_model.load_true_samples(tk, step=60, batch_size=50, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3,
                            pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=[''])
epochs = 2
gan_model.train(epochs=epochs, save_model_step=1, save_sample_step=1,
                print_batch=True, print_batch_step=20, print_epoch=True, print_epoch_step=1,
                warmup_steps=500, disc_lr=0.0001,
                optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                notes_latent_path=notes_latent_path, time_latent_path=time_latent_path,
                notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path, time_gen_path=time_gen_path,
                notes_disc_path=notes_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
                result_path=result_path,
                train_ntlatent=False, train_tmlatent=True, train_ntemb=False,
                train_ntgen=False, train_tmgen=False, train_disc=True,
                save_notes_ltnt=False, save_time_ltnt=True, save_notes_emb=False,
                save_notes_gen=False, save_time_gen=False, save_disc=False, max_to_keep=5,
                load_disc=False, disc_reinit_loss_thres=0.1, nt_ltnt_uniform=True, tm_ltnt_uniform=False)
# Todo: switch load_disc=True and save_disc=True once generator loss is lowered


# train on notes latent -------------------------------------------------
out_seq_len = 64  # 128
mode_ = 'notes'
gan_model = GAN(strt_token_id=15001, out_notes_pool_size=15002, embed_dim=256, n_heads=4, max_pos=800,
                time_features=3, fc_activation="relu",
                g_encoder_layers=2, g_decoder_layers=2, g_fc_layers=3, g_norm_epsilon=1e-6,
                g_embedding_dropout_rate=0.2, g_transformer_dropout_rate=0.2,
                d_kernel_size=3, d_encoder_layers=2, d_decoder_layers=2, d_fc_layers=3, d_norm_epsilon=1e-6,
                d_transformer_dropout_rate=0.2,
                notes_latent_nlayers=4, notes_latent_dim_base=4, time_latent_nlayers=4, out_seq_len=out_seq_len,
                mode_=mode_)
gan_model.load_true_samples(tk, step=60, batch_size=30, vel_norm=64.0, tmps_norm=0.12, dur_norm=1.3,
                            pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['noc'])  # Todo!!!!!
epochs = 3
gan_model.train(epochs=epochs, save_model_step=1, save_sample_step=1,
                print_batch=True, print_batch_step=2, print_epoch=True, print_epoch_step=1,
                warmup_steps=500, disc_lr=0.00001,
                optmzr=lambda lr: tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                notes_latent_path=notes_latent_path, time_latent_path=time_latent_path,
                notes_emb_path=notes_emb_path, notes_gen_path=notes_gen_path, time_gen_path=time_gen_path,
                notes_disc_path=notes_disc_path, time_disc_path=time_disc_path, combine_disc_path=combine_disc_path,
                result_path=result_path,
                train_ntlatent=True, train_tmlatent=False, train_ntemb=False,
                train_ntgen=False, train_tmgen=False, train_disc=True,
                save_notes_ltnt=True, save_time_ltnt=False, save_notes_emb=False,
                save_notes_gen=False, save_time_gen=False, save_disc=True, max_to_keep=5,
                load_disc=False, disc_reinit_loss_thres=0.1, nt_ltnt_uniform=True, tm_ltnt_uniform=False)

# Todo: switch load_disc=True and save_disc=True once generator loss is lowered




