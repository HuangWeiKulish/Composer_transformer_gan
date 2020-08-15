import numpy as np
import string
import mido
import glob
import os
import itertools
import pickle as pkl
from scipy import sparse


class Conversion:

    @staticmethod
    def switch_note(last_state, note, velocity, on_=True):
        # piano has 88 notes, which corresponding to note id 21 to 108, any note out of the id range will be ignored
        result = [0] * 88 if last_state is None else last_state.copy()
        if 21 <= note <= 108:
            result[note-21] = velocity if on_ else 0
        return result

    @staticmethod
    def msg2dict(msg):
        result = dict()
        if 'note_on' in msg:
            on_ = True
        elif 'note_off' in msg:
            on_ = False
        else:
            on_ = None
        result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
            str.maketrans({a: None for a in string.punctuation})))

        if on_ is not None:
            for k in ['note', 'velocity']:
                result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                    str.maketrans({a: None for a in string.punctuation})))
        return [result, on_]

    @staticmethod
    def get_new_state(new_msg, last_state):
        new_msg, on_ = Conversion.msg2dict(str(new_msg))
        new_state = Conversion.switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) \
            if on_ is not None else last_state
        return [new_state, new_msg['time']]

    @staticmethod
    def track2seq(track):
        # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
        result = []
        last_state, last_time = Conversion.get_new_state(str(track[0]), [0]*88)
        for i in range(1, len(track)):
            new_state, new_time = Conversion.get_new_state(track[i], last_state)
            if new_time > 0:
                result += [last_state]*new_time
            last_state, last_time = new_state, new_time
        return result

    @staticmethod
    def mid2arry(mid, min_msg_pct=0.1):
        tracks_len = [len(tr) for tr in mid.tracks]
        min_n_msg = max(tracks_len) * min_msg_pct
        # convert each track to nested list
        all_arys = []
        for i in range(len(mid.tracks)):
            if len(mid.tracks[i]) > min_n_msg:
                ary_i = Conversion.track2seq(mid.tracks[i])
                all_arys.append(ary_i)
        # make all nested list the same length
        max_len = max([len(ary) for ary in all_arys])
        for i in range(len(all_arys)):
            if len(all_arys[i]) < max_len:
                all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
        all_arys = np.array(all_arys)
        all_arys = all_arys.max(axis=0)
        # trim: remove consecutive 0s in the beginning and at the end
        sums = all_arys.sum(axis=1)
        ends = np.where(sums > 0)[0]
        return all_arys[min(ends): max(ends)]

    @staticmethod
    def arry2mid(ary, tempo=500000):
        # get the difference
        new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
        changes = new_ary[1:] - new_ary[:-1]
        # create a midi file with an empty track
        mid_new = mido.MidiFile()
        track = mido.MidiTrack()
        mid_new.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        # add difference in the empty track
        last_time = 0
        for ch in changes:
            if set(ch) == {0}:  # no change
                last_time += 1
            else:
                on_notes = np.where(ch > 0)[0]
                on_notes_vol = ch[on_notes]
                off_notes = np.where(ch < 0)[0]
                first_ = True
                for n, v in zip(on_notes, on_notes_vol):
                    new_time = last_time if first_ else 0
                    track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
                    first_ = False
                for n in off_notes:
                    new_time = last_time if first_ else 0
                    track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
                    first_ = False
                last_time = 0
        return mid_new


class DataPreparation:

    @staticmethod
    def get_all_filenames(filepath_list=['/Users/Wei/Desktop/piano_classic/Chopin_array'], name_substr_list=['noct']):
        file_names = []
        for filepath, name_substr in itertools.product(filepath_list, name_substr_list):
            file_names += glob.glob(os.path.join(filepath, '*' + name_substr + '*.pkl'))
        return file_names

    @staticmethod
    def get_notes_duration(x):
        # x is 1d array (length, ), each value in x represents a specific note id (from 1 to 88)
        # 0 means all notes are off
        # this function returns 2d array [[note1, duration1], [note2, duration2], ...]
        notes, dur = [], []
        cnt = 1
        x_init = x[0]
        for x_i in x[1:]:
            if x_i == x_init:
                cnt += 1
            else:
                dur.append(cnt)
                notes.append(x_init)
                x_init = x_i
                cnt = 1
        notes.append(x_init)
        dur.append(cnt)
        return np.array([notes, dur], dtype=object).T

    @staticmethod
    def token_encode(notes_i):
        # notes_i is 1d array (88,), with values represent note id
        result = '_'.join(notes_i[notes_i != 0].astype(str))
        return result if result != '' else 'p'  # 'p' for pause

    @staticmethod
    def token_decode(str_i):
        # str_i is string encoding of notes_i, e.g., '17_44_48_53', 'p' (pause)
        # this function decode str_i to binary array (88, )
        result = np.zeros((88,))
        if str_i != 'p':  # 'p' for pause
            tmp = str_i.split('_')
            indx = np.array(tmp).astype(int)
            result[indx] = 1
        return result

    @staticmethod
    def get_melody_array(x):
        # x is binary 2d array (length, 88)
        x_ = np.multiply(x, range(1, 89))
        x_ = [DataPreparation.token_encode(x_[i]) for i in range(len(x_))]
        # summarize notes and time duration of each note:
        # dim: (length, 2): [[notes1, duration1], [notes2, duration2], ...]
        result = DataPreparation.get_notes_duration(x_)
        return result

    @staticmethod
    def recover_array(x):
        # x_melody: dim = (length, 2): [[notes1, duration1], [notes2, duration2], ...]
        notes, dur = x.T
        result = []
        for n_i, d_i in zip(notes, dur):
            result += [DataPreparation.token_decode(n_i).tolist()] * int(d_i)
        return np.array(result)

    @staticmethod
    def cut_array(x, length, step, thres):
        result = []
        i = 0
        while length + i < x.shape[0]:
            result.append(x[i: (i + length)].tolist())
            i += step
        # if not to waste the last part of music
        if length + i - x.shape[0] >= thres:
            result.append(x[x.shape[0]-length:].tolist())
        return result

    @staticmethod
    def batch_preprocessing(in_seq_len, out_seq_len, step=60,
                            filepath_list=['/Users/Wei/Desktop/piano_classic/Chopin_array'], name_substr_list=['noct']):
        all_file_names = DataPreparation.get_all_filenames(
            filepath_list=filepath_list, name_substr_list=name_substr_list)
        x_in, x_tar = [], []
        for filepath in all_file_names:
            ary = pkl.load(open(filepath, 'rb'))  # file.shape = (length, 2)
            length = in_seq_len+out_seq_len
            ary = DataPreparation.cut_array(ary, length, step, int(step/3))  # (n_samples, length, 2)
            ary = np.array(ary, dtype=object)
            x_in += ary[:, :in_seq_len, :].tolist()
            x_tar += ary[:, in_seq_len:, :].tolist()
        return np.array(x_in, dtype=object), np.array(x_tar, dtype=object)


def process_midi(midifile_path='/Users/Wei/Desktop/piano_classic/Chopin',
                 save_path='/Users/Wei/Desktop/piano_classic/Chopin_array', to_sparse=True):
    all_mid_names = glob.glob(os.path.join(midifile_path, '*.mid'))
    for i, mid_name in enumerate(all_mid_names):
        if 'concerto' not in mid_name.lower():
            try:
                mid = mido.MidiFile(mid_name, clip=True)
                mid_array = Conversion.mid2arry(mid)
                mid_array = np.where(mid_array > 0, 1, 0)  # change to binary
                mid_converted = sparse.csr_matrix(mid_array) if to_sparse \
                    else DataPreparation.get_melody_array(mid_array)
                pkl.dump(mid_converted,
                         open(os.path.join(save_path, os.path.basename(mid_name).replace('.mid', '.pkl')), 'wb'))
            except:
                pass
            print(i)





"""
process_midi(midifile_path='/Users/Wei/Desktop/piano_classic/Chopin',
             save_path='/Users/Wei/Desktop/piano_classic/Chopin_array', to_sparse=False)
             
tmp = pkl.load(open('/Users/Wei/Desktop/piano_classic/Chopin_array/valse_70_3_(c)dery.pkl', 'rb'))
np.where(tmp=='p')

length = in_seq_len+out_seq_len
ary = DataPreparation.cut_array(tmp, length, step, int(step/3))  # (n_samples, length, 2)
ary = np.array(ary, dtype=object)
x_train = ary[:, :in_seq_len, :]
x_test = ary[:, in_seq_len:, :]




file = pkl.load(open(filepath, 'rb')).toarray()  # file.shape = (length, 88)
file = np.where(file > 0, 1, 0)  # change to binary
ary = DataPreparation.get_melody_array(file)  # array (length_original, 2)
length = in_seq_len+out_seq_len
ary = DataPreparation.cut_array(ary, length, step, int(step/3))  # (n_samples, length, 2)
ary = np.array(ary, dtype=object)
x_train += ary[:, :in_seq_len, :].tolist()
x_test += ary[:, in_seq_len:, :].tolist()
train_data = DataPreparation.
"""



"""
import matplotlib.pyplot as plt
import os

midifile_path = '/Users/Wei/Desktop/piano_classic/Chopin'
mid = mido. MidiFile(os.path.join(midifile_path, 'concerto_2_21_3_(c)finley.mid'), clip=True)
tmp = Conversion.mid2arry(mid)

tmp = tmp[:1000]
plt.plot(range(tmp.shape[0]), np.multiply(np.where(tmp > 0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
plt.title("nocturne_27_2_(c)inoue.mid")
plt.show()

mid_new = Conversion.arry2mid(tmp, 545455)
mid_new.save('mid_new.mid')

tst = Conversion.track2seq(mid.tracks[2])
np.where(np.array(tst).sum(axis=1)>0)  # 288

for m in mid.tracks[2][:50]:
    print(m)
"""

"""
filepath = '/Users/Wei/Desktop/piano_classic/Chopin_array/valse_70_2_(c)dery.pkl'
import pickle as pkl
filename = '/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer/result/50_0.pkl'
file = pkl.load(open(filename, 'rb')).toarray()*80
np.unique(file)

mid_new = Conversion.arry2mid(file.astype(int), 50)
mid_new.save('sample.mid')
"""


