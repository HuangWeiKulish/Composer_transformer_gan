import os
import preprocess
import pickle as pkl
import gensim


# get data ------------------------------------------------------------------
def get_data(pths='/Users/Wei/Desktop/midi_train/arry_modified'):
    all_file_names = preprocess.DataPreparation.get_all_filenames(pths=pths, name_substr_list=[''])
    result = []
    for filepath in all_file_names:
        ary = pkl.load(open(filepath, 'rb'))  # file.shape = (length, 2)
        result.append(ary[:, 0].tolist())
    return result

data = get_data(pths='/Users/Wei/Desktop/midi_train/arry_modified')


# skip-gram ------------------------------------------------------------------
embed_dim = 16
ch_embedder = gensim.models.Word2Vec(data, size=embed_dim, window=5, min_count=1, negative=20, iter=100)

model_path = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/chords_embedder'
ch_embedder.save(os.path.join(model_path, "chords_embedder.model"))


# continue training:
# ch_embedder = gensim.models.Word2Vec.load("word2vec.model")
# model.train(new_data, total_examples=1, epochs=1)

# check ------------------------------------------------------------------
# ch_embedder.wv.similar_by_word('64')
# ch_embedder.wv.most_similar(positive=['64'], negative=['32'])

# ch = embedder.wv.index2word  # chords list
# ch_embedder[ch_embedder.wv.vocab]  # chords vectors
