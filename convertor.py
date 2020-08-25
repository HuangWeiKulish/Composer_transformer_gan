import numpy as np
import string
import mido
import glob
import os
import itertools
import pickle as pkl
from scipy import sparse
import copy


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
        notes_ = copy.deepcopy(notes)
        # get all starting ticks
        all_notes = [[-1, -1, -1, -1, -1, -1, -1]]  # notes, index, velocity, start_tick, end_tick, start_time, end_time
        for k in notes_.keys():
            if len(notes_[k]) > 0:
                all_notes = np.append(
                    all_notes,
                    np.column_stack([[k]*len(notes_[k]), range(len(notes_[k])), np.array(notes_[k])]),
                    axis=0)
        all_notes = np.array(all_notes)[1:]
        return all_notes  # np 2d array, columns: [notes, index, velocity, start_tick, end_tick, start_time, end_time]

    @staticmethod
    def align_start(notes, thres_startgap=0.3, thres_pctdur=0.5):
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

    # @staticmethod
    # def align_end(notes, thres_endtgap=0.5, thres_pctdur=0.5):
    #     # notes: {21: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...],  ...,
    #     #         108: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...]}
    #     # all_notes: np 2d array, columns: [notes, index, velocity, start_tick, end_tick, start_time, end_time]
    #     notes_ = copy.deepcopy(notes)
    #
    #     # all_notes_: np 2d array, columns: [notes, index, velocity, start_tick, end_tick, start_time, end_time]
    #     all_notes_ = Conversion.combine_notes_info(notes)
    #     # all_notes_: 6 columns: [notes, index, strt_tick, end_tick, strt_time, end_time]
    #     all_notes_ = all_notes_[:, [0, 1, 3, 4, 5, 6]]
    #     all_notes_ = all_notes_[all_notes_[:, 3].argsort()]  # sort according to end_tk ascending
    #
    #     # modify notes' end_tick and end_time
    #     while len(all_notes_) > 0:
    #         end_max = all_notes_[0, 5] + thres_endtgap
    #         ids_included = np.where(all_notes_[:, 5] <= end_max)[0]
    #         if len(ids_included) > 1:
    #             nt_included = all_notes_[ids_included]
    #             new_tk = int(nt_included[:, 3].mean())
    #             new_tm = nt_included[:, 5].mean()
    #             for nt_, id_, s_tk, e_tk, s_tm, e_tm in nt_included:
    #                 # only modify end time if the distance to move is smaller than thres_pctdur * duration
    #                 if abs(e_tm-new_tm) <= thres_pctdur * abs(e_tm - s_tm):
    #                     id_ = int(id_)
    #                     notes_[nt_][id_][2] = new_tk
    #                     notes_[nt_][id_][4] = new_tm
    #             all_notes_ = all_notes_[max(ids_included):]
    #         else:
    #             all_notes_ = all_notes_[1:]
    #     return notes_

    @staticmethod
    def combine_notes_same_on_off(all_notes, thres_pctdur=0.4, thres_s=0.5, strick=True):
        # all_notes: np 2d array, columns: [notes, index, velocity, start_tick, end_tick, start_time, end_time]
        all_notes_ = all_notes[:, [0, 2, 3, 4, 5, 6]]  # [notes, velocity, start_tick, end_tick, start_time, end_time]
        all_notes_ = all_notes_[all_notes_[:, 2].argsort()]  # sort according to start_tick ascending
        result = [[{int(all_notes_[0][0])}] + all_notes_[0][1:].tolist()]
        for nt, vl, stk, etk, stm, etm in all_notes_[1:]:
            nt_set_0, vl_0, stk_0, etk_0, stm_0, etm_0 = result[-1]
            # combine if the two note_set start at the same time,
            #   and one end later than another for no more than max(thres_pctdur * note_set1_duration, thres_s)
            if stk == stk_0:
                if strick:
                    if etk == etk_0:
                        result[-1] = [nt_set_0.union({int(nt)}), max(vl, vl_0), stk, etk_0, stm, etm_0]
                else:
                    if abs(etm-etm_0) <= max(thres_pctdur * abs(etm_0-stm_0), thres_s):
                        result[-1] = [nt_set_0.union({int(nt)}), max(vl, vl_0), stk, etk_0, stm, etm_0]
            else:
                result.append([{int(nt)}, vl, stk, etk, stm, etm])
        return result

    @staticmethod
    def off_notes_earlier(all_notes, thres_endlate=0.4, thres_noteset_pctdur=0.8):
        # all_notes: [[notes_set, velocity, strt_tk, end_tk, strt_tm, end_tm], ...]
        all_notes_ = np.array(all_notes)
        # sort according to start tick ascending, and then end tick ascending
        all_notes_ = all_notes_[np.lexsort((all_notes_[:, 3], all_notes_[:, 2]))]
        result = [all_notes_[0].tolist()]
        all_notes_ = all_notes_[1:]
        while len(all_notes_) > 0:
            nts0, v0, strt_tk0, end_tk0, strt_tm0, end_tm0 = result[-1]
            dur0 = end_tm0 - strt_tm0
            nts_, v_, strt_tk_, end_tk_, strt_tm_, end_tm_ = all_notes_[0]
            if end_tm0 <= end_tm_:
                # 1. note_set1 ends before or at the same time as note_set2 ends:
                #   remove the overlap part of note_set1, if
                #   overlap_duration <= thres_noteset_pctdur * note_set1_duration
                if abs(end_tm_ - end_tm0) <= thres_noteset_pctdur * dur0:
                    result[-1] = [nts0, v0, strt_tk0, strt_tk_, strt_tm0, strt_tm_]
            else:
                # 2. note_set1 ends after note_set2 ends:
                # remove the overlap part of note_set1, if:
                #   note_set1 duration is long, and the ending part has very weak sound
                #   (the remove point start is calculated based on velocity and note_id)
                pass




            result.append([nts_, v_, strt_tk_, end_tk_, strt_tm_, end_tm_])
            all_notes_ = all_notes_[1:]
        return all_notes_

    @staticmethod
    def midinfo(mid, default_tempo=500000, notes_pool_reduction=True,
                thres_strt_gap=0.3, thres_strt_pctdur=0.5,  thres_end_gap=0.5, thres_end_pctdur=0.5,
                thres_noteset_pctdur=0.3):
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

        if notes_pool_reduction:
            # align start: --------------------------------------------------------------------------------------------
            #   if several notes start shortly one after another, replace a note's start time with mean start time when:
            #       1. time difference between note's original start time and mean start time is shorter
            #           than thres_strt_pctdur * notes_duration;
            #       2. time range between the earliest note and the latest note is shorter than thres_strt_gap (s)
            notes = Conversion.align_start(
                notes, thres_startgap=thres_strt_gap, thres_pctdur=thres_strt_pctdur)

            # align end: ----------------------------------------------------------------------------------------------
            #   if several notes end shortly one after another, replace a note's end time with mean end time when:
            #       1. time difference between note's original end time and mean end time is shorter
            #           than thres_end_pctdur * notes_duration;
            #       2. time range between the earliest note and the latest note is shorter than thres_end_gap (s)
            # notes = Conversion.align_end(
            #     notes, thres_endtgap=thres_end_gap, thres_pctdur=thres_end_pctdur)

            # combine notes with the same start and end as set, then  -------------------------------------------------
            #   if 2 note sets overlap, and note_set1 starts before note_set2 starts,
            #       1. remove the overlap part of note_set1, if:
            #           note_set1 ends before or at the same time as note_set2 ends,
            #           and overlap_duration <= thres_noteset_pctdur * note_set1_duration
            #       2. note_set1 duration is long, and the ending part has very weak sound
            #           (the remove point start is calculated based on velocity and note_id)
            all_notes = Conversion.combine_notes_info(notes)
            all_notes = Conversion.combine_notes_same_on_off(all_notes)




            # Todo: Conversion.combine_notes_same_on_off one more time in the end

            # all_notes.shape






            # string encode

            """
            # check
            for k in notes.keys():
                for x in notes[k]:
                    if (x[1] >= x[2]) | (x[3] >= x[4]):
                    #if (x[1] >= x[2]):
                        print('error')
                        print(k)
                        print(x)
            """




    # @staticmethod
    # def sets_encode_notes(nt, string_only=False):
    #     # returns a function of {notes set: velocity} if string_only is False
    #     #   else return string encode (example: '3_15', it means piano key 3 and key 15 are on,
    #     #       '' means everything is off)
    #     locs = np.where(nt > 0)[0]
    #     result = '_'.join(locs.astype(int).astype(str).tolist()) \
    #         if string_only else {k: v for k, v in zip(locs, nt[locs])}
    #     return result
    #
    # @staticmethod
    # def sets_encode_array(result_arys, result_time, string_only=False):
    #     # result_arys: (length, 88): 2d array containing velocity of each note
    #     # result_time: (length,): time duration of each tick
    #     # this function returns 2d array [[notes_velocity_dict1, duration1], [notes_velocity_dict2, duration2], ...]
    #     assert len(result_arys) == len(result_time)
    #     nt0, tm0 = Conversion.sets_encode_notes(result_arys[0], string_only), result_time[0]
    #     notes, dur = [], []
    #     for nt, tm in zip(result_arys[1:], result_time[1:]):
    #         nt = Conversion.sets_encode_notes(nt, string_only)
    #         if nt == nt0:
    #             tm0 += tm
    #         else:
    #             dur.append(tm0)
    #             notes.append(nt0)
    #             nt0, tm0 = nt, tm
    #     notes.append(nt0)
    #     dur.append(tm0)
    #     return np.array([notes, dur], dtype=object).T



    # @staticmethod
    # def arry2mid(ary, tempo=500000, velocity=70):
    #     # get the difference
    #     new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
    #     changes = new_ary[1:] - new_ary[:-1]
    #     # create a midi file with an empty track
    #     mid_new = mido.MidiFile()
    #     track = mido.MidiTrack()
    #     mid_new.tracks.append(track)
    #     track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    #     # add difference in the empty track
    #     last_time = 0
    #     for ch in changes:
    #         if set(ch) == {0}:  # no change
    #             last_time += 1
    #         else:
    #             on_notes = np.where(ch > 0)[0]
    #             on_notes_vol = ch[on_notes]
    #             off_notes = np.where(ch < 0)[0]
    #             first_ = True
    #             for n, v in zip(on_notes, on_notes_vol):
    #                 new_time = last_time if first_ else 0
    #                 track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
    #                 first_ = False
    #             for n in off_notes:
    #                 new_time = last_time if first_ else 0
    #                 track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
    #                 first_ = False
    #             last_time = 0
    #     return mid_new



# class DataPreparation:
#
#     @staticmethod
#     def get_all_filenames(filepath_list=['/Users/Wei/Desktop/piano_classic/Chopin_array'], name_substr_list=['noct']):
#         file_names = []
#         for filepath, name_substr in itertools.product(filepath_list, name_substr_list):
#             file_names += glob.glob(os.path.join(filepath, '*' + name_substr + '*.pkl'))
#         return file_names
#
#     @staticmethod
#     def get_notes_duration(x):
#         # x is 1d array (length, ), each value in x represents a specific note id (from 1 to 88)
#         # 0 means all notes are off
#         # this function returns 2d array [[note1, duration1], [note2, duration2], ...]
#         notes, dur = [], []
#         cnt = 1
#         x_init = x[0]
#         for x_i in x[1:]:
#             if x_i == x_init:
#                 cnt += 1
#             else:
#                 dur.append(cnt)
#                 notes.append(x_init)
#                 x_init = x_i
#                 cnt = 1
#         notes.append(x_init)
#         dur.append(cnt)
#         return np.array([notes, dur], dtype=object).T
#
#     @staticmethod
#     def token_encode(notes_i):
#         # notes_i is 1d array (88,), with values represent note id
#         result = '_'.join(notes_i[notes_i != 0].astype(str))
#         return result if result != '' else 'p'  # 'p' for pause
#
#     @staticmethod
#     def token_decode(str_i):
#         # str_i is string encoding of notes_i, e.g., '17_44_48_53', 'p' (pause)
#         # this function decode str_i to binary array (88, )
#         result = np.zeros((88,))
#         if str_i != 'p':  # 'p' for pause
#             tmp = str_i.split('_')
#             indx = np.array(tmp).astype(int)
#             result[indx] = 1
#         return result
#
#     @staticmethod
#     def get_melody_array(x):
#         # x is binary 2d array (length, 88)
#         x_ = np.multiply(x, range(1, 89))
#         x_ = [DataPreparation.token_encode(x_[i]) for i in range(len(x_))]
#         # summarize notes and time duration of each note:
#         # dim: (length, 2): [[notes1, duration1], [notes2, duration2], ...]
#         result = DataPreparation.get_notes_duration(x_)
#         return result
#
#     @staticmethod
#     def recover_array(x):
#         # x_melody: dim = (length, 2): [[notes1, duration1], [notes2, duration2], ...]
#         notes, dur = x.T
#         result = []
#         for n_i, d_i in zip(notes, dur):
#             result += [DataPreparation.token_decode(n_i).tolist()] * int(d_i)
#         return np.array(result)
#
#     @staticmethod
#     def cut_array(x, length, step, thres):
#         result = []
#         i = 0
#         while length + i < x.shape[0]:
#             result.append(x[i: (i + length)].tolist())
#             i += step
#         # if not to waste the last part of music
#         if length + i - x.shape[0] >= thres:
#             result.append(x[x.shape[0]-length:].tolist())
#         return result
#
#     @staticmethod
#     def batch_preprocessing(in_seq_len, out_seq_len, step=60,
#                             filepath_list=['/Users/Wei/Desktop/piano_classic/Chopin_array'], name_substr_list=['noct']):
#         all_file_names = DataPreparation.get_all_filenames(
#             filepath_list=filepath_list, name_substr_list=name_substr_list)
#         x_in, x_tar = [], []
#         for filepath in all_file_names:
#             ary = pkl.load(open(filepath, 'rb'))  # file.shape = (length, 2)
#             length = in_seq_len+out_seq_len
#             ary = DataPreparation.cut_array(ary, length, step, int(step/3))  # (n_samples, length, 2)
#             ary = np.array(ary, dtype=object)
#             x_in += ary[:, :in_seq_len, :].tolist()
#             x_tar += ary[:, in_seq_len:, :].tolist()
#         return np.array(x_in, dtype=object), np.array(x_tar, dtype=object)
#
#
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
#




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


