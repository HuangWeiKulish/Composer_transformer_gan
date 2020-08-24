import numpy as np
import string
import mido
import glob
import os
import itertools
import pickle as pkl
from scipy import sparse
import copy


# source: https://www.piano-keyboard-guide.com/keyboard-chords.html
# chords_dict = {
#     # major
#     'C major': 'C E G',
#     'C# major': 'C# E# G#',
#     'D major': 'D F# A',
#     'Eb major': 'Eb G Bb',
#     'E major': 'E G# B',
#     'F major': 'F A C',
#     'F# major': 'F# A# C#',
#     'G major': 'G B D',
#     'Ab major': 'Ab C Eb',
#     'A major': 'A C# E',
#     'Bb major': 'Bb D F',
#     'B major': 'B D# F#',
#     # minor
#     'C minor': 'C Eb G',
#     'C# minor': 'C# E G#',
#     'D minor': 'D F A',
#     'Eb minor': 'Eb Gb Bb',
#     'E minor': 'E G B',
#     'F minor': 'F Ab C',
#     'F# minor': 'F# A C#',
#     'G minor': 'G Bb D',
#     'Ab minor': 'Ab B Eb',
#     'A minor': 'A C E',
#     'Bb minor': 'Bb Db F',
#     'B minor': 'B D F#',
#     # diminished
#     'C diminished': 'C Eb Gb',
#     'C# diminished': 'C# E G',
#     'D diminished': 'D F Ab',
#     'Eb diminished': 'Eb Gb A',
#     'E diminished': 'E G Bb',
#     'F diminished': 'F Ab B',
#     'F# diminished': 'F# A C',
#     'G diminished': 'G Bb Db',
#     'Ab diminished': 'Ab Cb D',
#     'A diminished': 'A C Eb',
#     'Bb diminished': 'Bb Db E',
#     'B diminished': 'B D F',
#     # major 7th
#     'C major seventh': 'C E G B',
#     'C# major seventh': 'C# F G# C',
#     'D major seventh': 'D F# A C#',
#     'Eb major seventh': 'Eb G Bb D',
#     'E major seventh': 'E G# B D#',
#     'F major seventh': 'F A C E',
#     'F# major seventh': 'F# A# C# F',
#     'G major seventh': 'G B D F#',
#     'Ab major seventh': 'Ab C Eb G',
#     'A major seventh': 'A C# E G#',
#     'Bb major seventh': 'Bb D F A',
#     'B major seventh': 'B D# F# A#',
#     # dominant 7th
#     'C dominant seventh': 'C E G Bb',
#     'C# dominant seventh': 'C# E# G# B',
#     'D dominant seventh': 'D F# A C',
#     'Eb dominant seventh': 'Eb G Bb Db',
#     'E dominant seventh': 'E G# B D',
#     'F dominant seventh': 'F A C Eb',
#     'F# dominant seventh': 'F# A# C# E',
#     'G dominant seventh': 'G B D F',
#     'Ab dominant seventh': 'Ab C Eb Gb',
#     'A dominant seventh': 'A C# E G',
#     'Bb dominant seventh': 'Bb D F Ab',
#     'B dominant seventh': 'B D# F# A',
#     # minor 7th
#     'C minor seventh': 'C Eb G Bb',
#     'C# minor seventh': 'C# E G# B',
#     'D minor seventh': 'D F A C',
#     'Eb minor seventh': 'Eb Gb Bb Db',
#     'E minor seventh': 'E G B D',
#     'F minor seventh': 'F Ab C Eb',
#     'F# minor seventh': 'F# A C# E',
#     'G minor seventh': 'G Bb D F',
#     'Ab minor seventh': 'Ab Cb Eb Gb',
#     'A minor seventh': 'A C E G',
#     'Bb minor seventh': 'Bb Db F Ab',
#     'B minor seventh': 'B D F# A',
#     # minor 7th flat chord
#     'C minor seventh flat five': 'C Eb Gb Bb',
#     'C# minor seventh flat five': 'C# E G B',
#     'D minor seventh flat five': 'D F Ab C',
#     'Eb minor seventh flat five': 'Eb Gb A Db',
#     'E minor seventh flat five': 'E G Bb D',
#     'F minor seventh flat five': 'F Ab B Eb',
#     'F# minor seventh flat five': 'F# A C E',
#     'G minor seventh flat five': 'G Bb Db F',
#     'Ab minor seventh flat five': 'Ab Cb D Gb',
#     'A minor seventh flat five': 'A C Eb G',
#     'Bb minor seventh flat five': 'Bb Db E Ab',
#     'B minor seventh flat five': 'B D F A',
# }
#
# notes_dict = {
#     'B#': 0,
#     'C': 0,
#     'C#': 1,
#     'Db': 1,
#     'D': 2,
#     'D#': 3,
#     'Eb': 3,
#     'E': 4,
#     'Fb': 4,
#     'F': 5,
#     'E#': 5,
#     'F#': 6,
#     'Gb': 6,
#     'G': 7,
#     'G#': 8,
#     'Ab': 8,
#     'A': 9,
#     'A#': 10,
#     'Bb': 10,
#     'B': 11,
#     'Cb': 11,
# }
#
# chords_dict_numb = {
#     k: '_'.join(np.array(np.sort(np.array([notes_dict[n] for n in chords_dict[k].split(' ')]))).astype(str))
#     for k in chords_dict.keys()
# }


# --------------------------------------
# Todo: track always starts from 0? can different track starts from different time?


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
            if msg['tempo'] > 0:
                tempos.append([msg['tempo'], time_passed])
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
                tmp = tmp[tmp[:, 1].argsort()].tolist()
                notes_[k].append(tmp[0])
                for v_, s_, e_ in tmp[1:]:
                    v0, s0, e0 = notes_[k][-1]
                    if e0 < s_:
                        notes_[k].append([v_, s_, e_])
                    elif s_ <= e0 < e_:
                        notes_[k][-1] = [v0, s0, e_]
                    else:  # e0 >= e_
                        notes_[k][-1] = [v0, s0, e0]
        return notes_

    @staticmethod
    def add_time_value(notes, all_strt, time_values):
        # and add time value of start and end ticks
        notes_ = copy.deepcopy(notes)
        all_strt_tm = []
        for tm, st, nd in time_values:
            if st == nd:
                nts = all_strt[all_strt[:, -1] >= st]
            else:  # st < nd
                nts = all_strt[(all_strt[:, -1] >= st) & (all_strt[:, -1] < nd)]
        return notes_

    @staticmethod
    def combine_notes_info(notes):
        # notes: {21: [[velocity1, strt1, end1], ...], ..., 108: [[velocity1, strt1, end1], ...]}
        notes_ = copy.deepcopy(notes)
        # get all starting ticks
        all_strt = [[-1, -1, -1, -1, -1]]  # notes, index, start_tick, end_tick, end_tick
        for k in notes_.keys():
            if len(notes_[k]) > 0:
                all_strt = np.append(
                    all_strt,
                    np.column_stack([[k]*len(notes_[k]), range(len(notes_[k])), np.array(notes_[k])]),
                    axis=0)
        all_strt = np.array(all_strt)[1:]
        return all_strt  # np 2d array, columns: [note_id1, id_, velocity, strt, end]

    @staticmethod
    def align_start(notes, all_strt, time_values, thres_startgap=0.3, thres_pctdur=0.5):
        # notes: {21: [[velocity1, strt1, end1], ...], ..., 108: [[velocity1, strt1, end1], ...]}
        # all_strt: np 2d array, columns: [note_id1, id_, velocity, strt, end]
        notes_ = copy.deepcopy(notes)
        all_strt_ = all_strt[:, [0, 1, 3, 4]].copy()  # 3 columns: [note_id1, id_, strt, end]
        all_strt_ = all_strt_[all_strt_[:, 2].argsort()]  # sort according to strt
        # modify notes' start_tick
        for tm, st, nd in time_values:
            if st == nd:
                nts = all_strt_[all_strt_[:, -1] >= st]
            else:  # st < nd
                nts = all_strt_[(all_strt_[:, -1] >= st) & (all_strt_[:, -1] < nd)]
            if len(nts) > 0:
                while len(nts) > 0:
                    strt_max = nts[0, 2] + thres_startgap / tm
                    ids_included = np.where(nts[:, 2] <= strt_max)[0]
                    if len(ids_included) > 1:
                        nt_included = nts[ids_included]
                        new_tm = int(nt_included[:, 2].mean())
                        for nt_, id_, s_, e_ in nt_included:
                            # only modify start time if the distance to move is smaller than thres_pctdur * duration
                            if abs(s_-new_tm) <= thres_pctdur * abs(e_ - s_):
                                notes_[nt_][id_][1] = new_tm
                        nts = nts[max(ids_included):]
                    else:
                        nts = nts[1:]
        return notes_

    @staticmethod
    def align_end(notes, all_strt, time_values, thres_startgap=0.3, thres_pctdur=0.5):
        # notes: {21: [[velocity1, strt1, end1], ...], ..., 108: [[velocity1, strt1, end1], ...]}
        # all_strt: np 2d array, columns: [note_id1, id_, velocity, strt, end]
        notes_ = copy.deepcopy(notes)
        all_strt_ = all_strt[:, [0, 1, 3, 4]].copy()  # 3 columns: [note_id1, id_, strt, end]
        all_strt_ = all_strt_[all_strt_[:, 3].argsort()[::-1]]  # sort according to end, descending

        # modify notes' end_tick
        all_strt_tm = []
        for tm, st, nd in time_values:
            if st == nd:
                nts = all_strt_[all_strt_[:, -1] >= st]
            else:  # st < nd
                nts = all_strt_[(all_strt_[:, -1] >= st) & (all_strt_[:, -1] < nd)]





        if len(nts) > 0:
            while len(nts) > 0:
                strt_max = nts[0, 2] + thres_startgap / tm
                ids_included = np.where(nts[:, 2] <= strt_max)[0]
                if len(ids_included) > 1:
                    nt_included = nts[ids_included]
                    new_tm = int(nt_included[:, 2].mean())
                    for nt_, id_, s_, e_ in nt_included:
                        # only modify start time if the distance to move is smaller than thres_pctdur * duration
                        if abs(s_-new_tm) <= thres_pctdur * abs(e_ - s_):
                            notes_[nt_][id_][1] = new_tm
                    nts = nts[max(ids_included):]
                else:
                    nts = nts[1:]
        return notes_

    @staticmethod
    def off_notes(notes, all_strt, thres_startgap=0.3, thres_pctdur=0.5):
        # notes: {21: [[velocity1, strt1, end1], ...], ..., 108: [[velocity1, strt1, end1], ...]}
        # all_strt: np 2d array, columns: [note_id1, id_, velocity, strt, end]



        pass





    @staticmethod
    def midinfo(mid, default_tempo=500000, notes_pool_reduction=True,
                thres_strt_startgap=0.3, thres_strt_pctdur=0.5):
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
        all_strt = Conversion.combine_notes_info(notes)
        notes = Conversion.clean_notes(notes)
        notes = Conversion.add_time_value(notes, all_strt, time_values)


        if notes_pool_reduction:
            # all_strt: np 2d array, columns: [note_id1, id_, velocity, strt, end]
            all_strt = Conversion.combine_notes_info(notes)
            # align start:
            #   if several notes start shortly one after another, replace a note's start time with mean start time when:
            #       1. time difference between note's original start time and mean start time is shorter
            #           than thres_strt_pctdur * notes_duration;
            #       2. time range between the earliest note and the latest note is shorter than thres_strt_startgap (s)
            notes = Conversion.align_start(
                notes, all_strt, time_values, thres_startgap=thres_strt_startgap, thres_pctdur=thres_strt_pctdur)
            all_strt = Conversion.combine_notes_info(notes)




            #   if 2 notes are off one after another shortly,
            #       and the off time difference is shorter than max_pct_timegap_end of both notes duration:
            #           combine the 2 notes end time, and the end time will be the medium end time of both
            #   if the overlap time is shorter than max_pct_timegap_ovl of the earlier notes duration:,
            #       and also shorter than the later notes duration:
            #           replace the overlap with the second note
            # off previous notes if they are not started together



















    @staticmethod
    def sets_encode_notes(nt, string_only=False):
        # returns a function of {notes set: velocity} if string_only is False
        #   else return string encode (example: '3_15', it means piano key 3 and key 15 are on,
        #       '' means everything is off)
        locs = np.where(nt > 0)[0]
        result = '_'.join(locs.astype(int).astype(str).tolist()) \
            if string_only else {k: v for k, v in zip(locs, nt[locs])}
        return result

    @staticmethod
    def sets_encode_array(result_arys, result_time, string_only=False):
        # result_arys: (length, 88): 2d array containing velocity of each note
        # result_time: (length,): time duration of each tick
        # this function returns 2d array [[notes_velocity_dict1, duration1], [notes_velocity_dict2, duration2], ...]
        assert len(result_arys) == len(result_time)
        nt0, tm0 = Conversion.sets_encode_notes(result_arys[0], string_only), result_time[0]
        notes, dur = [], []
        for nt, tm in zip(result_arys[1:], result_time[1:]):
            nt = Conversion.sets_encode_notes(nt, string_only)
            if nt == nt0:
                tm0 += tm
            else:
                dur.append(tm0)
                notes.append(nt0)
                nt0, tm0 = nt, tm
        notes.append(nt0)
        dur.append(tm0)
        return np.array([notes, dur], dtype=object).T



    @staticmethod
    def arry2mid(ary, tempo=500000, velocity=70):
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

midifile_path = '/Users/Wei/Desktop/selected/chopin'
mid = mido.MidiFile(os.path.join(midifile_path, 'ballade_op_23_no_1_a_(nc)smythe.mid'), clip=True)





ary = Conversion.mid2arry(mid)






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


