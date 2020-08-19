from transformer_gan.gan_composer_transformer import GAN, DataPreparation
import os, glob
import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from convertor import Conversion

result_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/result'
model_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/models'

g_model_path_name = os.path.join(model_path, 'gmodel.h5')
gan = GAN(g_input_dim=100, g_fc_dim=100, g_fltr_dims=(20, 10, 5, 3, 1, 1), g_knl_dims=(5, 7, 9, 11, 13, 15),
          g_up_dims=(2, 2, 2, 2, 2, 2), g_bn_momentum=0.6, g_noise_std=0.2, g_final_fltr=1, g_final_knl=5,
          g_model_path_name=g_model_path_name, g_loaded_trainable=True, g_loss=None,
          d_fltr_dims=(1, 1, 3, 5, 10, 20), d_knl_dims=(15, 13, 11, 9, 7, 5), d_down_dims=(2, 2, 2, 2, 2, 2),
          d_bn_momentum=0.6, d_model_path_name=None, d_loaded_trainable=True, d_lr=0.0001, d_clipnorm=1.0,
          d_loss='mse', d_metrics=['accuracy'],
          f_lr=0.01, f_clipnorm=1.0, print_model=True, latent_mean=0.0, latent_std=1.1)

# load data =================================
filepath_list = ['/Users/Wei/Desktop/piano_classic/Chopin_array']  # '/Users/Wei/Desktop/piano_classic/Beethoven/sonata_variations'
name_substr_list = ['sonat']  # 'variation'

x = DataPreparation.batch_preprocessing(
    length=gan.gmodel.output.shape[1], step=200, filepath_list=filepath_list, name_substr_list=name_substr_list)
print(x.dtype)
print(x.shape)
# print(np.unique(x))


# pre-train gmodel ==========================
# gan.gmodel.compile(loss=loss_func, optimizer=Adam(lr=0.01, clipnorm=1.0), metrics=['mse'])
# for i in range(4):
#     gan.gmodel.fit(d, x, epochs=10, verbose=1, validation_split=0.1, batch_size=20)
#     gan.gmodel.save(os.path.join(model_path, 'gmodel.h5'))
#
# # predict
# ind = -1
# pred = gan.gmodel.predict(d[[[ind]]])
#
# plt.figure(figsize=(20, 5))
# pred = np.rint(np.squeeze(pred, axis=0))  # round it to integer
# plt.plot(range(pred.shape[0]), np.multiply(pred, range(1, 89)), marker='.',
#          markersize=1, linestyle='')
# plt.show()
#
# # true
# plt.figure(figsize=(20, 5))
# plt.plot(range(x[ind].shape[0]), np.multiply(x[ind], range(1, 89)), marker='.',
#          markersize=1, linestyle='')
# plt.show()


# train gan =================================

for i in range(1000):
    gan.train(x, n_epoch=50, n_samples=50, save_step=10, n_save=1, verbose=True, save_pic=True,
              file_save_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/result',
              model_save_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/models')
    gan = GAN(g_input_dim=100, g_fc_dim=100, g_fltr_dims=(20, 10, 5, 3, 1, 1), g_knl_dims=(5, 7, 9, 11, 13, 15),
              g_up_dims=(2, 2, 2, 2, 2, 2), g_bn_momentum=0.6, g_noise_std=0.2, g_final_fltr=1, g_final_knl=5,
              g_model_path_name=os.path.join(model_path, 'gmodel.h5'), g_loaded_trainable=True, g_loss=None,
              d_fltr_dims=(1, 1, 3, 5, 10, 20), d_knl_dims=(15, 13, 11, 9, 7, 5), d_down_dims=(2, 2, 2, 2, 2, 2),
              d_bn_momentum=0.6, d_model_path_name=None, d_loaded_trainable=True, d_lr=0.0001, d_clipnorm=1.0,
              d_loss='mse', d_metrics=['accuracy'],
              f_lr=0.01, f_clipnorm=1.0, print_model=False, latent_mean=0.0, latent_std=1.1)

# tmp = gan.gmodel.predict(DataPreparation.latant_vector(1, 100, mean_=0.0, std_=1.1))
# DataPreparation.recover_array(tmp).shape


# check =======================

filename = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/result/40_0.pkl'
file = pkl.load(open(filename, 'rb')).toarray().astype(int) * 80

plt.matshow(file[:1000].T)



np.unique(file)
mid_new = Conversion.arry2mid(file, 1)
mid_new.save('/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/sample.mid')

