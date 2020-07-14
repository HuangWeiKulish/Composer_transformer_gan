from gan_composer import GAN
import os, glob
import pickle as pkl

model = GAN(g_input_dim=500, g_filter_list=(8, 6, 4), g_strides_list=((2, 11), (2, 4), (1, 2)),
            g_kernel_list=(5, 4, 3), g_resnet_up_add_noise=False, g_upsample_add_noise=True,
            g_model_path_name=None, g_loaded_trainable=False,
            d_input_dim=2000, d_filter_list=(4, 6, 8), d_kernel_list=(3, 4, 5), d_pool=True, d_dropout=0.2,
            d_model_path_name=None, d_loaded_trainable=False,
            d_lr=0.01, d_clipnorm=1., f_lr=0.01, f_clipnorm=1., pixmax=127)
model.gmodel.summary()
model.dmodel.summary()


result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/result'
model_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer'

model.load_data(piece_length=2000, data_path='/Users/Wei/Desktop/piano_classic/Chopin_array', name_contain='*noct*')
model.train(n_epoch=500, n_samples=20, save_step=10, n_save=2, verbose=True, save_as_mid=False, save_pic=True,
            save_path=result_path)

# save models
all_models = [int(nm.replace('.h5', '').split('_')[-1]) for nm in glob.glob(os.path.join(model_path, 'gmodel_*.h5'))]
model.gmodel.save(os.path.join(model_path, 'gmodel_{}.h5'.format(max(all_models)+1 if len(all_models) > 0 else 0)))
model.dmodel.save(os.path.join(model_path, 'dmodel_{}.h5'.format(max(all_models)+1 if len(all_models) > 0 else 0)))

pkl.dump({'d_perform': model.d_perform,
          'f_perform': model.f_perform},
         open(os.path.join(model_path, 'performance_{}.pkl'.format(max(all_models)+1 if len(all_models) > 0 else 0))))



"""
model.full_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01, clipnorm=1.0))
model.dmodel.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01, clipnorm=1.0), metrics=['accuracy'])


n_samples=20
g_input_dim=500

fake_x = model.fake_samples(n_samples)
fake_x.shape
true_x = model.true_samples(n_samples=20)
true_x.shape
true_y, fake_y = np.ones((n_samples, 1), dtype=np.float32), np.zeros((n_samples, 1), dtype=np.float32)

# (None, 2000, 88, 1)
model.dmodel.trainable=True

model.dmodel.train_on_batch(true_x, true_y)
model.dmodel.summary()
latent_x = np.random.normal(0.0, 1.0, size=[n_samples*2, g_input_dim, 1, 1])
model.full_model.train_on_batch(latent_x, np.zeros((n_samples*2, 1), dtype=np.float32))

"""
