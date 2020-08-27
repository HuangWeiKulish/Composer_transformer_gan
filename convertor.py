import numpy as np
import string
import mido
import mido.midifiles.meta as meta
del meta._META_SPECS[0x59]
del meta._META_SPEC_BY_TYPE['key_signature']
import os
import itertools
import pickle as pkl
import copy
import glob
import tensorflow as tf
import json


class Conversion:

    @staticmethod
    def msg2dict(msg):
        result = dict()
        result['tempo'] = int(msg[msg.rfind('tempo'):].split(' ')[0].split('=')[1].translate(
            str.maketrans({a: None for a in string.punctuation}))) if 'tempo' in msg else 0
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
            if result['velocity'] == 0:
                on_ = False
        return [result, on_]

    @staticmethod
    def track2seq(track):
        # piano has 88 notes, which corresponding to note id 21 to 108, any note out of the id range will be ignored
        notes = {k: [] for k in range(21, 109)}
        tempos = []
        time_passed = 0
        last_note = None
        for i in range(1, len(track)):
            msg, on_ = Conversion.msg2dict(str(track[i]))
            # time here means number of ticks (the time duration of each tick depends on tempo, and tick_per_sec)
            time_passed += msg['time']
            if on_ is not None:
                if on_:
                    if 21 <= msg['note'] <= 108:
                        notes[msg['note']].append([msg['velocity'], time_passed, time_passed])
                else:  # on_ is False
                    if 21 <= msg['note'] <= 108:
                        notes[msg['note']][-1][-1] = time_passed  # modify the last off time
                last_note = msg['note']
            if msg['tempo'] > 0:
                tempos.append([msg['tempo'], time_passed])
        if last_note is not None:
            notes[last_note][-1][-1] = time_passed  # modify the final off time
        return notes, tempos

    @staticmethod
    def get_time_value(tempos, ticks_per_beat=120, default_tempo=500000):
        # example: tempos = [[400000, 3], [300000, 5]]: 2 columns: [tempo, start tick]
        if len(tempos) > 0:
            tempos = np.array(tempos)
            tempos = tempos[tempos[:, 1].argsort()]
            result = [[mido.tick2second(1, ticks_per_beat, default_tempo), 0, 0]] if tempos[0][-1] > 0 else []
            for tmp, st in tempos:
                if len(result) > 0:
                    result[-1][-1] = st
                result += [[mido.tick2second(1, ticks_per_beat, tmp), st, st]]  # time per tick
        else:
            result = [[mido.tick2second(1, ticks_per_beat, default_tempo), 0, 0]]
        # result: 3 columns: [time_per_tick, start tick, end tick]  the last end tick is the same as the last start tick
        return result

    @staticmethod
    def clean_notes(notes):
        # notes: {21: [[velocity1, strt1, end1], ...], ..., 108: [[velocity1, strt1, end1], ...]}
        # connect overlapped same notes together
        notes_ = dict()
        for k in notes.keys():
            notes_[k] = []
            if len(notes[k]) > 0:
                tmp = np.array(notes[k])
                tmp = tmp[tmp[:, 1].argsort()]
                if tmp[0, 1] >= tmp[0, 2]:  # if start tick is not smaller than end tick:
                    tmp[0, 2] += tmp[0, 2] - tmp[0, 1] + 1
                notes_[k].append(tmp[0].tolist())
                for v_, s_, e_ in tmp[1:]:
                    v0, s0, e0 = notes_[k][-1]
                    if s0 == s_:
                        notes_[k][-1] = [v0, s0, e0]
                    else:  # s0 < s_
                        if s_ < e_:  # if start tick is smaller than end tick:
                            if e0 < s_:
                                notes_[k].append([v_, s_, e_])
                            elif s_ <= e0 < e_:
                                notes_[k][-1] = [v0, s0, s_]
                                notes_[k].append([v_, s_+1, e_])
                            else:  # e0 >= e_
                                notes_[k][-1] = [v0, s0, e0]
        return notes_

    @staticmethod
    def add_time_value(notes, time_values):
        # notes: {21: [[velocity1, strt1, end1], ...], ..., 108: [[velocity1, strt1, end1], ...]}
        # time_values: 3 columns: [time_per_tick, start tick, end tick]
        # This function adds time value of start and end ticks
        # return: {21: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...],
        #          ...,
        #          108: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...]}
        notes_ = dict()
        for k in notes.keys():
            v = np.array(notes[k])
            notes_[k] = []
            if len(v) > 0:
                for tm, st, nd in time_values:
                    if st == nd:
                        nts = v[v[:, 1] >= st]
                    else:  # st < nd
                        nts = v[(v[:, 1] >= st) & (v[:, 1] < nd)]
                    notes_[k] += np.column_stack([nts, nts[:, 1] * tm, nts[:, 2] * tm]).tolist()
        return notes_

    @staticmethod
    def combine_notes_info(notes):
        # notes: {21: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...],  ...,
        #         108: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...]}
        # get all starting ticks
        all_notes = [[-1, -1, -1, -1, -1, -1, -1]]
        for k in notes.keys():
            if len(notes[k]) > 0:
                all_notes = np.append(
                    all_notes,
                    np.column_stack([[k]*len(notes[k]), range(len(notes[k])), np.array(notes[k])]),
                    axis=0)
        all_notes = np.array(all_notes)[1:]
        return all_notes  # np 2d array, columns: [notes, index, velocity, start_tick, end_tick, start_time, end_time]

    @staticmethod
    def align_start(notes, thres_startgap=0.2, thres_pctdur=0.3):
        # notes: {21: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...],  ...,
        #         108: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...]}
        notes_ = copy.deepcopy(notes)

        # all_notes_: np 2d array, columns: [notes, index, velocity, start_tick, end_tick, start_time, end_time]
        all_notes_ = Conversion.combine_notes_info(notes)
        # all_notes_: 6 columns: [notes, index, strt_tick, end_tick, strt_time, end_time]
        all_notes_ = all_notes_[:, [0, 1, 3, 4, 5, 6]]
        all_notes_ = all_notes_[all_notes_[:, 2].argsort()]  # sort according to strt_tick ascending

        # modify notes' start_tick and start_time
        while len(all_notes_) > 0:
            strt_max = all_notes_[0, 4] + thres_startgap
            ids_included = np.where(all_notes_[:, 4] <= strt_max)[0]
            if len(ids_included) > 1:
                nt_included = all_notes_[ids_included]
                new_tk = int(nt_included[:, 2].mean())
                new_tm = nt_included[:, 4].mean()
                for nt_, id_, s_tk, e_tk, s_tm, e_tm in nt_included:
                    # only modify start time if the distance to move is smaller than thres_pctdur * duration
                    if abs(s_tm-new_tm) <= thres_pctdur * abs(e_tm - s_tm):
                        id_ = int(id_)
                        notes_[nt_][id_][1] = new_tk
                        notes_[nt_][id_][3] = new_tm
                all_notes_ = all_notes_[max(ids_included):]
            else:
                all_notes_ = all_notes_[1:]
        return notes_

    @staticmethod
    def combine_notes_same_on_off(notes, thres_pctdur=0.4, thres_s=0.5, clean=True):
        # all_notes: np 2d array, columns: [notes, velocity, start_tick, end_tick, start_time, end_time]
        all_notes = Conversion.combine_notes_info(notes)[:, [0, 2, 3, 4, 5, 6]].astype(object)
        all_notes_ = all_notes[all_notes[:, 2].argsort()]  # sort according to start_tick ascending
        result = []
        if clean:
            all_strt_tm = np.unique(all_notes_[:, 4])
            for st in all_strt_tm:
                # combine if the two note_set start at the same time,
                nt_stm = all_notes_[all_notes_[:, 4] == st]
                ends = nt_stm[:, 5]
                ends_mid = (ends.max()+ends.min()) / 2.0
                tolerance = max(thres_pctdur * ends_mid, thres_s)
                if max(ends_mid - min(ends), max(ends) - ends_mid) <= tolerance:
                    # end time difference is within tolerance: combine
                    result.append([set(nt_stm[:, 0].astype(int)), nt_stm[:, 1].mean(),  # notes_set, velocity
                                   nt_stm[0, 2], int(nt_stm[:, 3].mean()),              # start_tick, end_tick
                                   st, np.median(nt_stm[:, 5])])                        # start_time, end_time
                else:
                    # end time difference is too large: split into 2
                    nt_stm_1 = nt_stm[nt_stm[:, 5] <= ends_mid]
                    result.append([set(nt_stm_1[:, 0].astype(int)), nt_stm_1[:, 1].mean(),  # notes_set, velocity
                                   nt_stm_1[0, 2], int(nt_stm_1[:, 3].mean()),              # start_tick, end_tick
                                   st, np.median(nt_stm_1[:, 5])])                          # start_time, end_time
                    nt_stm_2 = nt_stm[nt_stm[:, 5] > ends_mid]
                    result.append([set(nt_stm_2[:, 0].astype(int)), nt_stm_2[:, 1].mean(),  # notes_set, velocity
                                   nt_stm_2[0, 2], int(nt_stm_2[:, 3].mean()),              # start_tick, end_tick
                                   st, np.median(nt_stm_2[:, 5])])                          # start_time, end_time
        else:
            all_notes_[:, 0] = [{x} for x in all_notes_[:, 0].astype(int)]
        return np.array(result)

    @staticmethod
    def chords_limit(all_notes, max=10):
        # all_notes: np 2d array, columns: [notes, velocity, start_tick, end_tick, start_time, end_time]
        all_notes_ = all_notes.copy()
        for i in range(len(all_notes_)):
            if len(all_notes_[i][0]) > max:
                l_ = list(all_notes_[i][0])
                l_.sort()
                all_notes_[i][0] = set(l_[(len(all_notes_[i][0])-max):])
        return all_notes_

    @staticmethod
    def encode_notes(all_notes, use_time=True):
        # all_notes: 2d array: [notes, velocity, start_tick, end_tick, start_time, end_time]
        all_notes = all_notes[all_notes[:, 2].argsort()]  # sort according to start_tick ascending
        # encode notes to string
        notes_str = ['_'.join(np.array(list(x))[np.array(list(x)).argsort()].astype('str'))
                     for x in all_notes[:, 0]]
        velocity = all_notes[:, 1]
        start = all_notes[:, 4] if use_time else all_notes[:, 2]
        end = all_notes[:, 5] if use_time else all_notes[:, 3]
        notes_duration = end - start
        # Todo: the first time_passed_since_last_note is not useful, change to 0?
        time_passed_since_last_note = start - np.array([0]+start[:-1].tolist())
        return np.array([notes_str, velocity, time_passed_since_last_note, notes_duration]).T

    @staticmethod
    def mid2arry(mid, default_tempo=500000, clean=True, thres_strt_gap=0.2, thres_strt_pctdur=0.3,
                 thres_noteset_pctdur=0.3, thres_noteset_s=0.5, chords_nmax=10, use_time=True):
        # convert each track to nested list
        notes = {k: [] for k in range(21, 109)}
        tempos = []
        for i in range(len(mid.tracks)):
            notes_i, tempo_i = Conversion.track2seq(mid.tracks[i])
            for k in notes.keys():
                if len(notes_i[k]) > 0:
                    notes[k] += notes_i[k]
            if len(tempo_i) > 0:
                tempos += tempo_i
        # time_values: 3 columns: [time_per_tick, start tick, end tick]
        #   the last end tick is the same as the last start tick
        time_values = Conversion.get_time_value(tempos, mid.ticks_per_beat, default_tempo)
        # notes: {21: [[velocity1, strt1, end1], ...], ..., 108: [[velocity1, strt1, end1], ...]}
        # connect overlapped same notes together, and add time value of start and end ticks
        notes = Conversion.clean_notes(notes)
        notes = Conversion.add_time_value(notes, time_values)
        # notes: {21: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...],  ...,
        #         108: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...]}

        if clean:
            # align start: --------------------------------------------------------------------------------------------
            #   if several notes start shortly one after another, replace a note's start time with mean start time when:
            #       1. time difference between note's original start time and mean start time is shorter
            #           than thres_strt_pctdur * notes_duration;
            #       2. time range between the earliest note and the latest note is shorter than thres_strt_gap (s)
            notes = Conversion.align_start(
                notes, thres_startgap=thres_strt_gap, thres_pctdur=thres_strt_pctdur)

            # combine notes: ------------------------------------------------------------------------------------------
            #   1. align notes which are on at almost the same time
            #   2. then, for the notes on at the same time, allow them to end at 2 different times at most
            all_notes = Conversion.combine_notes_same_on_off(
                notes, thres_pctdur=thres_noteset_pctdur, thres_s=thres_noteset_s, clean=True)
            all_notes = Conversion.chords_limit(all_notes, max=chords_nmax)
        else:
            all_notes = Conversion.combine_notes_same_on_off(
                notes, thres_pctdur=0.4, thres_s=0.5, clean=False)

        # encode -----------------------------------
        all_notes = Conversion.encode_notes(all_notes, use_time=use_time)
        return all_notes

    @staticmethod
    def arry2mid(ary, tempo=500000, ticks_per_beat=120, use_time=True):
        # ary: 2d array: [notes_str, velocity, time_passed_since_last_note, notes_duration], in start sequence order
        notes_str = ary[:, 0]
        velocity = ary[:, 1]
        time_passed_since_last_note = ary[:, 2]
        notes_duration = ary[:, 3]

        # calculate start tick and end tick
        start = np.cumsum(time_passed_since_last_note)
        end = start + notes_duration
        if use_time:  # convert time to number of ticks
            start = np.array([mido.second2tick(x, ticks_per_beat=ticks_per_beat, tempo=tempo) for x in start])
            end = np.array([mido.second2tick(x, ticks_per_beat=ticks_per_beat, tempo=tempo) for x in end])

        # summarize information
        ary_new = []
        for nts, vl, st, ed in zip(notes_str, velocity, start, end):
            nts_ = nts.split('_')
            for nt_i in nts_:
                # notes on message
                ary_new.append(['note_on', int(nt_i), int(np.rint(vl)), int(np.rint(st))])
                ary_new.append(['note_off', int(nt_i), int(np.rint(vl)), int(np.rint(ed))])
        ary_new = np.array(ary_new, dtype=object)
        ary_new = ary_new[ary_new[:, 3].argsort()]

        # create a midi file with an empty track
        mid_new = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid_new.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        # add information in the track
        last_time = 0
        for m, nt, vl, tm in ary_new:
            track.append(mido.Message(m, note=nt, velocity=vl, time=tm-last_time))
            last_time = tm
        return mid_new


def batch_convert_midi2arry(midifile_path='/Users/Wei/Desktop/midi_train/midi',
                            array_path='/Users/Wei/Desktop/midi_train/arry',
                            default_tempo=500000, clean=True, thres_strt_gap=0.2, thres_strt_pctdur=0.2,
                            thres_noteset_pctdur=0.3, thres_noteset_s=0.5, use_time=True, print_progress=True):
    midifile_subdir = [f.path for f in os.scandir(midifile_path) if f.is_dir()]
    array_subdir = [f.path for f in os.scandir(array_path) if f.is_dir()]
    array_makedir = [os.path.join(array_path, os.path.basename(x)) for x in midifile_subdir]
    print('start to process')

    for ary_sd, md_sd in zip(array_makedir, midifile_subdir):
        print()
        if ary_sd not in array_subdir:
            os.mkdir(ary_sd)
        midi_files = os.listdir(md_sd)
        ary_files = os.listdir(ary_sd)
        for mdf in midi_files:
            filenm, extn = os.path.splitext(mdf)
            if (mdf not in ary_files) & (extn.lower() == '.mid'):
                try:
                    mid_ = mido.MidiFile(os.path.join(md_sd, mdf), clip=True)
                    ary_ = Conversion.mid2arry(
                        mid_, default_tempo=default_tempo, clean=clean, thres_strt_gap=thres_strt_gap,
                        thres_strt_pctdur=thres_strt_pctdur, thres_noteset_pctdur=thres_noteset_pctdur,
                        thres_noteset_s=thres_noteset_s, use_time=use_time)
                    pkl.dump(ary_, open(os.path.join(ary_sd, filenm+'.pkl'), 'wb'))
                    if print_progress:
                        print(os.path.join(ary_sd, filenm+'.pkl'))
                except:
                    pass

"""
batch_convert_midi2arry(midifile_path='/Users/Wei/Desktop/midi_train/midi',
                            array_path='/Users/Wei/Desktop/midi_train/arry',
                            default_tempo=500000, clean=True, thres_strt_gap=0.2, thres_strt_pctdur=0.2,
                            thres_noteset_pctdur=0.3, thres_noteset_s=0.5, use_time=True, print_progress=True)
"""


def notes_indexing(
        array_path='/Users/Wei/Desktop/midi_train/arry',
        tk_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_dict.pkl',
        print_num=True):
    try:
        tk = pkl.load(open(tk_path, 'rb'))
    except:
        tk = tf.keras.preprocessing.text.Tokenizer(filters='')
    array_subdir = [f.path for f in os.scandir(array_path) if f.is_dir()]
    for sb in array_subdir:
        nms = os.listdir(sb)
        for nm in nms:
            filenm, extn = os.path.splitext(nm)
            if extn.lower() == '.pkl':
                ary = pkl.load(open(os.path.join(sb, nm), 'rb'))
                tk.fit_on_texts(ary[:, 0].tolist())
                if print_num:
                    print('Number notes: {}'.format(len(json.loads(tk.get_config()['word_counts']))))
    pkl.dump(tk, open(tk_path, 'wb'))



    # ary = all_notes.copy()


def get_infrequent_notes_str(tk, lim_cnt=5, lim_pctl=90):
    nts_cnt = np.array(list(json.loads(tk.get_config()['word_counts']).items()), dtype=object)
    cnt_max = max(np.percentile(nts_cnt[:, 1], lim_pctl), lim_cnt)
    return nts_cnt[nts_cnt[:, 1] <= cnt_max, 0]


def infrequent_notes_str_replacement(tk, lim_cnt=5, lim_pctl=90):
    # replace infrequent notes string with more frequent notes string
    infq_nt = get_infrequent_notes_str(tk, lim_cnt=lim_cnt, lim_pctl=lim_pctl)
    


"""
notes_indexing(
        array_path='/Users/Wei/Desktop/midi_train/arry', 
        tk_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_dict.pkl',
        print_num=True)

tk = pkl.load(open(tk_path, 'rb'))

import matplotlib.pyplot as plt
tmp = np.array(list(json.loads(tk.get_config()['word_counts']).items()), dtype=object)
len(tmp)  # 64700

np.median(tmp[:, 1])
len(tmp[tmp[:, 1] >= np.median(tmp[:, 1])])  # 38304

np.mean(tmp[:, 1])
len(tmp[tmp[:, 1] >= np.mean(tmp[:, 1])])  # 3389

pctl = 80
np.percentile(tmp[:, 1], pctl)
len(tmp[tmp[:, 1] >= np.percentile(tmp[:, 1], pctl)])  # 14989

tmp[tmp[:, 1]<2, 0]



plt.hist(tmp[:, 1], bins=500)
plt.xlim(0, 5000)
plt.show()


# 64700  
"""


class DataPreparation:

    @staticmethod
    def get_all_filenames(filepath_list=['/Users/Wei/Desktop/piano_classic/Chopin_array'], name_substr_list=['']):
        file_names = []
        for filepath, name_substr in itertools.product(filepath_list, name_substr_list):
            file_names += glob.glob(os.path.join(filepath, '*' + name_substr + '*.pkl'))
        return file_names

    @staticmethod
    def cut_array(x, length, step, thres):
        # x: np 2d array: columns [notes_str, velocity, time_passed_since_last_note, notes_duration]
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








# tk.get_config()







# def process_midi(midifile_path='/Users/Wei/Desktop/piano_classic/Chopin',
#                  save_path='/Users/Wei/Desktop/piano_classic/Chopin_array', to_sparse=True):
#     all_mid_names = glob.glob(os.path.join(midifile_path, '*.mid'))
#     for i, mid_name in enumerate(all_mid_names):
#         if 'concerto' not in mid_name.lower():
#             try:
#                 mid = mido.MidiFile(mid_name, clip=True)
#                 mid_array = Conversion.mid2arry(mid)
#                 mid_array = np.where(mid_array > 0, 1, 0)  # change to binary
#                 mid_converted = sparse.csr_matrix(mid_array) if to_sparse \
#                     else DataPreparation.get_melody_array(mid_array)
#                 pkl.dump(mid_converted,
#                          open(os.path.join(save_path, os.path.basename(mid_name).replace('.mid', '.pkl')), 'wb'))
#             except:
#                 pass
#             print(i)


"""

import os

midifile_path = '/Users/Wei/Desktop/midi_train/midi/chopin'
mid = mido.MidiFile(os.path.join(midifile_path, 'ballade_op_23_no_1_a_(nc)smythe.mid'), clip=True)
ary = Conversion.mid2arry(mid, default_tempo=500000, clean=True, thres_strt_gap=0.2, thres_strt_pctdur=0.2,
                 thres_noteset_pctdur=0.3, thres_noteset_s=0.5, use_time=True)
new_mid = Conversion.arry2mid(ary, tempo=500000, ticks_per_beat=120, use_time=True)
new_mid.save('/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/new_mid.mid')
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


