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
        # tempos: [tempo, start from tick]
        # example: tempos = [[400000, 3], [300000, 5]]: 2 columns: [tempo, start tick]
        if len(tempos) > 0:
            tempos = np.array(tempos)
            tempos = tempos[tempos[:, 1].argsort()]
            result = [[mido.tick2second(1, ticks_per_beat, default_tempo), 0, 0]] if tempos[0][-1] > 0 else []
            for tmp, st in tempos:
                if len(result) > 0:
                    result[-1][-1] = st
                new_ = [mido.tick2second(1, ticks_per_beat, tmp), st, st]  # time per tick
                if len(result) > 0:
                    if new_ != result[-1]:
                        result += [new_]
                else:
                    result += [new_]
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
    def add_time_value(all_notes, time_values):
        # all_notes (np 2d array): [note_id, velocity, strt_tk, end_tk]
        # time_values: 3 columns: [time_per_tick, start tick, end tick]
        # This function adds time value of start and end ticks
        # return (np 2d array): [note_id, velocity, strt_tk, end_tk, strt_tm, end_tm]

        time_values_ = np.array(time_values)
        time_values_ = time_values_[time_values_[:, 1].argsort()]
        tm_passed = time_values_[:, 0] * np.array([0] + (time_values_[1:, 1] - time_values_[:-1, 1]).tolist())
        # [time_per_tick, start tick, end tick, time passed]
        time_values_ = np.column_stack([time_values_, np.cumsum(tm_passed)])

        result = []
        last_tm_ps, last_tk = 0, 0
        for tm, st, ed, tm_ps in time_values_:
            if st == ed:
                nts = all_notes[all_notes[:, 2] >= st]
            else:  # st < nd
                nts = all_notes[(all_notes[:, 2] >= st) & (all_notes[:, 2] < ed)]
            strt_tm = (nts[:, 2] - last_tk)*tm + last_tm_ps
            end_tm = (nts[:, 3] - last_tk)*tm + last_tm_ps
            result += np.column_stack([nts, strt_tm, end_tm]).tolist()
            last_tm_ps, last_tk = tm_ps, st
        return np.array(result, dtype=object)

    @staticmethod
    def combine_notes_info(notes):
        # notes: {21: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...],  ...,
        #         108: [[velocity1, strt1, end1, strt_tm1, end_tm1], ...]}
        # get all starting ticks
        dim = 5
        for k in notes.keys():
            if len(notes[k]) > 0:
                dim = np.array(notes[k]).shape[1]
                break
        all_notes = [[-1]*(dim+1)]
        for k in notes.keys():
            if len(notes[k]) > 0:
                all_notes = np.append(
                    all_notes,
                    #np.column_stack([[k]*len(notes[k]), range(len(notes[k])), np.array(notes[k])]),
                    np.column_stack([[k] * len(notes[k]), np.array(notes[k])]),
                    axis=0)
        all_notes = np.array(all_notes)[1:]
        return all_notes  # np 2d array, columns: [notes, index, velocity, start_tick, end_tick, start_time, end_time]

    @staticmethod
    def align_start(all_notes, thres_startgap=0.2, thres_pctdur=0.3):
        # all_notes (np 2d array): [note_id, velocity, strt_tk, end_tk, strt_tm, end_tm]
        # all_notes_ (np 2d array): [note_id, velocity, strt_tk, end_tk, strt_tm, end_tm, index]
        all_notes_ = np.column_stack([all_notes, range(len(all_notes))])
        note_id_col, velocity_col, strt_tk_col, end_tk_col, strt_tm_col, end_tm_col, index_col = range(7)
        all_notes_ = all_notes_[all_notes_[:, strt_tk_col].argsort()]  # sort according to strt_tick ascending

        # modify notes' start_tick and start_time
        last_id = 0
        while last_id < len(all_notes_):
            strt_tm0 = all_notes_[last_id, strt_tm_col]
            strt_max = strt_tm0 + thres_startgap
            ids_included = np.where((all_notes_[:, strt_tm_col] >= strt_tm0) &
                                    (all_notes_[:, strt_tm_col] < strt_max))[0]
            if len(ids_included) > 1:
                nt_included = all_notes_[ids_included]
                new_tk = int(nt_included[:, strt_tk_col].mean())
                new_tm = nt_included[:, strt_tm_col].mean()

                for nt_, vl_, s_tk, e_tk, s_tm, e_tm, id_ in nt_included:
                    # only modify start time if the distance to move is smaller than thres_pctdur * duration
                    if abs(s_tm - new_tm) <= thres_pctdur * abs(e_tm - s_tm):
                        id_ = int(id_)
                        all_notes_[id_, strt_tk_col] = new_tk
                        all_notes_[id_, strt_tm_col] = new_tm
                last_id = max(ids_included) + 1
            else:
                last_id += 1
        # all_notes_ (np 2d array): [note_id, velocity, strt_tk, end_tk, strt_tm, end_tm, index]
        return all_notes_

    @staticmethod
    def combine_notes_same_on_off(all_notes, thres_pctdur=0.4, thres_s=0.5, clean=True):
        # all_notes (np 2d array): [note_id, velocity, strt_tk, end_tk, strt_tm, end_tm, index]
        note_id_col, velocity_col, strt_tk_col, end_tk_col, strt_tm_col, end_tm_col, index_col = range(7)
        all_notes_ = all_notes[all_notes[:, strt_tk_col].argsort()]  # sort according to start_tick ascending
        result = []
        if clean:
            all_strt_tm = np.unique(all_notes_[:, strt_tm_col])
            for st in all_strt_tm:
                # combine if the two note_set start at the same time,
                nt_stm = all_notes_[all_notes_[:, strt_tm_col] == st]
                ends = nt_stm[:, end_tm_col]
                ends_mid = (ends.max()+ends.min()) / 2.0
                tolerance = max(thres_pctdur * ends_mid, thres_s)
                if max(ends_mid - min(ends), max(ends) - ends_mid) <= tolerance:
                    # end time difference is within tolerance: combine
                    result.append([
                        # notes_set, velocity
                        set(nt_stm[:, note_id_col].astype(int)), nt_stm[:, velocity_col].mean(),
                        # start_tick, end_tick
                        nt_stm[0, strt_tk_col], int(nt_stm[:, end_tk_col].mean()),
                        # start_time, end_time
                        st, np.median(nt_stm[:, end_tm_col])])

                else:
                    # end time difference is too large: split into 2
                    nt_stm_1 = nt_stm[nt_stm[:, end_tm_col] <= ends_mid]
                    result.append([
                        # notes_set, velocity
                        set(nt_stm_1[:, note_id_col].astype(int)), nt_stm_1[:, velocity_col].mean(),
                        # start_tick, end_tick
                        nt_stm_1[0, strt_tk_col], int(nt_stm_1[:, end_tk_col].mean()),
                        # start_time, end_time
                        st, np.median(nt_stm_1[:, end_tm_col])])
                    nt_stm_2 = nt_stm[nt_stm[:, end_tm_col] > ends_mid]
                    result.append([
                        # notes_set, velocity
                        set(nt_stm_2[:, note_id_col].astype(int)), nt_stm_2[:, velocity_col].mean(),
                        # start_tick, end_tick
                        nt_stm_2[0, strt_tk_col], int(nt_stm_2[:, end_tk_col].mean()),
                        # start_time, end_time
                        st, np.median(nt_stm_2[:, end_tm_col])])
            result = np.array(result)
        else:
            result = all_notes_[:, [note_id_col, velocity_col, strt_tk_col, end_tk_col, strt_tm_col, end_tm_col]].copy()
            result[:, note_id_col] = [{x} for x in all_notes_[:, note_id_col].astype(int)]
        return result

    # @staticmethod
    # def chords_limit(all_notes, max=10):
    #     # all_notes: np 2d array, columns: [notes, velocity, start_tick, end_tick, start_time, end_time]
    #     all_notes_ = all_notes.copy()
    #     for i in range(len(all_notes_)):
    #         if len(all_notes_[i][0]) > max:
    #             l_ = list(all_notes_[i][0])
    #             l_.sort()
    #             all_notes_[i][0] = set(l_[(len(all_notes_[i][0])-max):])
    #     return all_notes_

    @staticmethod
    def encode_notes(all_notes, use_time=True):
        # all_notes (np 2d array): [set(note_id1, ...), velocity, strt_tk, end_tk, strt_tm, end_tm, index]
        note_id_col, velocity_col, strt_tk_col, end_tk_col, strt_tm_col, end_tm_col = range(6)
        sort_col = strt_tm_col if use_time else strt_tk_col
        all_notes = all_notes[all_notes[:, sort_col].argsort()]  # sort according to start_tick ascending

        # encode notes to string
        notes_str = ['_'.join(np.array(list(x))[np.array(list(x)).argsort()].astype('str'))
                     for x in all_notes[:, 0]]
        velocity = all_notes[:, 1]
        start = all_notes[:, sort_col]
        end = all_notes[:, sort_col]
        notes_duration = end - start
        time_passed_since_last_note = start - np.array([0]+start[:-1].tolist())

        result = np.array([notes_str, velocity, time_passed_since_last_note, notes_duration]).T
        # Todo: can there be time_passed_since_last_note < 0???????

        return result

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

        # notes: {21: [[velocity1, strt_tk1, end_tk1], ...], ..., 108: [[velocity1, strt_tk1, end_tk1], ...]}
        # connect overlapped same notes together, and add time value of start and end ticks
        notes = Conversion.clean_notes(notes)
        # all_notes (np 2d array): [note_id, velocity, strt_tk, end_tk]
        all_notes = Conversion.combine_notes_info(notes)
        # all_notes (np 2d array): [note_id, velocity, strt_tk, end_tk, strt_tm, end_tm]
        all_notes = Conversion.add_time_value(all_notes, time_values)

        if clean:
            # align start: --------------------------------------------------------------------------------------------
            #   if several notes start shortly one after another, replace a note's start time with mean start time when:
            #       1. time difference between note's original start time and mean start time is shorter
            #           than thres_strt_pctdur * notes_duration;
            #       2. time range between the earliest note and the latest note is shorter than thres_strt_gap (s)
            # all_notes (np 2d array): [note_id, velocity, strt_tk, end_tk, strt_tm, end_tm, index]
            all_notes = Conversion.align_start(
                all_notes, thres_startgap=thres_strt_gap, thres_pctdur=thres_strt_pctdur)
            # all_notes (np 2d array): [note_id, velocity, strt_tk, end_tk, strt_tm, end_tm, index]

            # combine notes: ------------------------------------------------------------------------------------------
            #   1. align notes which are on at almost the same time
            #   2. then, for the notes on at the same time, allow them to end at 2 different times at most
            all_notes = Conversion.combine_notes_same_on_off(
                all_notes, thres_pctdur=thres_noteset_pctdur, thres_s=thres_noteset_s, clean=True)
            # all_notes (np 2d array): [note_id, velocity, strt_tk, end_tk, strt_tm, end_tm]
            # all_notes = Conversion.chords_limit(all_notes, max=chords_nmax)
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
                    #
                    #
                    # if len(np.where(ary_[:, 2] < 0)[0]) > 0:
                    #     print(os.path.join(md_sd, mdf))
                    #     print('negative cnt = {}'.format(len(np.where(ary_[:, 2] < 0)[0])))
                    #
                    #

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


"""
notes_indexing(
        array_path='/Users/Wei/Desktop/midi_train/arry', 
        tk_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_dict.pkl',
        print_num=True)
"""


class ReplaceNotes:

    @staticmethod
    def get_notes_frequency(tk):
        return np.array(list(json.loads(tk.get_config()['word_counts']).items()), dtype=object)

    @staticmethod
    def nts_cnt_str2set(nts_cnt):
        result = nts_cnt.copy()
        for i in range(len(result)):
            result[i, 0] = set(np.array(result[i, 0].split('_')).astype(int))
        return result

    @staticmethod
    def find_octave(x, higher=True, lower=True, highest_thres=None):
        # nt is int between 0 and 88,
        # this function fins all octaves of x, including itself
        # x is included in result, and highest_thres may also be included in result
        highest_thres = 88 if highest_thres is None else min(highest_thres, 88)
        result = []
        if lower:
            result += [x-12*i for i in range(int(x/12)+1)]
        if higher:
            result += [x + 12 * i for i in range(int((highest_thres - x) / 12) + 1)]
        return set(result)

    @staticmethod
    def split_notesset(notes_set):
        # example: notes_set={12, 24, 46, 73, 11, 56, 68} will be splitted into
        #   oct_sets = [{56, 68}, {12, 24}] and other_set = {11, 46, 73}
        oct_sets, other_set = [], set()
        tmp = notes_set.copy()
        while len(tmp) > 0:
            x = list(tmp)[0]
            x_oct = ReplaceNotes.find_octave(x, higher=True, lower=True, highest_thres=None)
            overlap = tmp.intersection(x_oct)
            if len(overlap) > 1:
                oct_sets.append(overlap)
            else:  # len(overlap) = 1
                other_set = other_set.union(overlap)
            tmp -= overlap
        return oct_sets, other_set

    @staticmethod
    def sample_list_sets(list_sets):
        # list_sets is a list of sets
        # sample at least one element from each set in list_sets, find out all possible samples
        result = []
        for s_ in list_sets:
            result.append(list(itertools.chain(*[itertools.combinations(s_, i) for i in range(1, len(s_))])))
        result_final = list(itertools.chain(*[list(itertools.product(*result[i:])) for i in range(len(result)-1)]))
        result_final = [set(itertools.chain(*x)) for x in result_final]
        result_final += [set(x) for x in itertools.chain(*result)]
        return result_final

    @staticmethod
    def find_octave_replacement(nt):
        #   for chord, the octave used as a replacement for lower notes cannot be higher than the highest note
        #       and the octave used as a replacement for the highest note can only be higher
        nt_ = nt.copy()
        highest_nt = max(nt)
        nt_.remove(highest_nt)
        # find octave for the highest notes
        octs = ReplaceNotes.find_octave(highest_nt, higher=True, lower=False, highest_thres=None) - {highest_nt}
        result = [nt_.union({x}) for x in octs]

        for x in nt:
            octs = ReplaceNotes.find_octave(x, higher=True, lower=True, highest_thres=highest_nt)
            octs.remove(x)
            for y in octs:
                if y not in nt:
                    nt_ = nt.copy()
                    nt_.remove(x)
                    nt_.add(y)
                    result.append(nt_)
        return result

    @staticmethod
    def rm_octave(nt, oct_sets, frq_nts_cnt):
        # rule 1: try to remove octave if exists
        rm = ReplaceNotes.sample_list_sets(oct_sets)
        new_nt = [nt - x for x in rm]
        nt_frq = []
        for nt_ in new_nt:
            tmp = frq_nts_cnt[frq_nts_cnt[:, 0] == nt_]
            if len(tmp) > 0:
                nt_frq += tmp.tolist()
        if len(nt_frq) > 0:
            # result: [original set, replacement set]
            return [nt, np.array(nt_frq)[np.array(nt_frq)[:, 1].argmax(), 0]]  # get the one with max frequency
        return None

    @staticmethod
    def replace_octave(nt, frq_nts_cnt):
        # Todo: the octave used to replace of lower notes cannot be higher than the highest note
        # rule 2. try to replace with notes which is n octaves (12*n) apart
        #   for chord, the octave used as a replacement for lower notes cannot be higher than the highest note
        #       and the octave used as a replacement for the highest note can only be higher
        new_nt = ReplaceNotes.find_octave_replacement(nt)
        nt_frq = []
        for nt_ in new_nt:
            tmp = frq_nts_cnt[frq_nts_cnt[:, 0] == nt_]
            if len(tmp) > 0:
                nt_frq += tmp.tolist()
        if len(nt_frq) > 0:
            # [original set, replacement set]
            return [nt, np.array(nt_frq)[np.array(nt_frq)[:, 1].argmax(), 0]]
        return None

    @staticmethod
    def subset_notes(nt, frq_nts_cnt):
        # rule 3. try to use subset of notes
        nt_sub = itertools.chain.from_iterable(itertools.combinations(nt, r) for r in range(len(nt)+1))
        result = []
        for nt_ in nt_sub:
            tmp = frq_nts_cnt[frq_nts_cnt[:, 0] == set(nt_)]
            if len(tmp) > 0:
                result += tmp.tolist()
        if len(result) > 0:
            return [nt, np.array(result)[np.array(result)[:, 1].argmax(), 0]]
        else:
            return None

    @staticmethod
    def get_note_octave(nt, frq_nts_cnt):
        nt_ = np.sort(list(nt))[::-1].tolist()
        while len(nt_) > 0:
            nt_h = nt_.pop(0)
            result = frq_nts_cnt[np.isin(
                frq_nts_cnt[:, 0],
                [{x} for x in ReplaceNotes.find_octave(nt_h, higher=True, lower=True, highest_thres=None)])]
            if len(result) > 0:
                return [nt, result[result[:, 1].argmax(), 0]]
        return None

    @staticmethod
    def find_notes_replacement(nt, frq_nts_cnt):
        # return result: [original set, replacement set]
        # rule 1: try to remove octave if exists
        oct_sets, other_set = ReplaceNotes.split_notesset(nt)
        if len(oct_sets) > 0:
            result = ReplaceNotes.rm_octave(nt, oct_sets, frq_nts_cnt)
            if result is not None:
                return result
        # rule 2. try to replace with notes which is n octaves (12*n) apart
        #   for chord, the octave used as a replacement lower notes cannot be higher than the highest note
        result = ReplaceNotes.replace_octave(nt, frq_nts_cnt)
        if result is not None:
            return result
        # rule 3. try to use subset of notes
        result = ReplaceNotes.subset_notes(nt, frq_nts_cnt)
        if result is not None:
            return result
        # rule 4: try to use the octave of the highest note first, followed by the next highest
        result = ReplaceNotes.get_note_octave(nt, frq_nts_cnt)
        if result is not None:
            return result
        return [nt, {65}]  # default

    @staticmethod
    def infrequent_notes_replacement(tk, keep_top_n=15000, show_progress=True, progress_step=100):
        # get a dictionary which matches infrequent notes string with frequent notes string
        nts_cnt = ReplaceNotes.get_notes_frequency(tk)
        nts_cnt = ReplaceNotes.nts_cnt_str2set(nts_cnt)
        print('number notes strings = {}'.format(len(nts_cnt)))

        nts_cnt = nts_cnt[nts_cnt[:, 1].argsort()[::-1]]
        frq_nts_cnt = nts_cnt[: keep_top_n]
        infq_nts = nts_cnt[keep_top_n:, 0]
        print('number frequent notes strings = {}'.format(len(frq_nts_cnt)))
        print('number infrequent notes strings = {}'.format(len(infq_nts)))

        result = []
        for i, nt in enumerate(infq_nts):
            result.append(ReplaceNotes.find_notes_replacement(nt, frq_nts_cnt))
            if show_progress:
                if (i+1) % progress_step == 0:
                    print(round((i+1) / len(infq_nts), 3))
        return np.array(result)

    @staticmethod
    def get_notes_replacement(notes_replacement):
        result = dict()
        for og, rp in notes_replacement:
            og = list(og)
            og.sort()
            og = '_'.join(np.array(og).astype(str))
            rp = list(rp)
            rp.sort()
            rp = '_'.join(np.array(rp).astype(str))
            result[og] = rp
        return result


"""
tk_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_dict.pkl'
tk = pkl.load(open(tk_path, 'rb'))

nts_cnt = ReplaceNotes.get_notes_frequency(tk)
nts_cnt = ReplaceNotes.nts_cnt_str2set(nts_cnt)
print(len(nts_cnt))
thres = np.percentile(nts_cnt[:, 1], 85)
print(thres)
print(len(nts_cnt[nts_cnt[:, 1] > thres]))

notes_replacement = ReplaceNotes.infrequent_notes_replacement(
    tk, keep_top_n=15000, show_progress=True, progress_step=100)
notes_replacement = ReplaceNotes.get_notes_replacement(notes_replacement)    
    
    
# save notes_replacement:
nt_rp_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_replacement.pkl'
pkl.dump(notes_replacement, open(nt_rp_path, 'wb'))
"""


def replace_infreq_nts_in_ary(ary, notes_replacement):
    result = ary.copy()
    for i in range(len(ary)):
        nt = result[i, 0]
        if nt in notes_replacement.keys():
            result[i, 0] = notes_replacement[nt]
    return result


def batch_replace_infreq_nts(
        notes_replacement, array_path='/Users/Wei/Desktop/midi_train/arry',
        array_mod_path='/Users/Wei/Desktop/midi_train/arry_modified', show_progress=True):
    array_subdir = [f.path for f in os.scandir(array_path) if f.is_dir()]
    array_mod_subdir = [f.path for f in os.scandir(array_mod_path) if f.is_dir()]
    array_makedir = [os.path.join(array_mod_path, os.path.basename(x)) for x in array_subdir]
    print('start to process')

    for ary_mod_sd, ary_sd in zip(array_makedir, array_subdir):
        if ary_mod_sd not in array_mod_subdir:
            os.mkdir(ary_mod_sd)
        ary_files = os.listdir(ary_sd)
        arymod_files = os.listdir(ary_mod_sd)
        for nm in ary_files:
            filenm, extn = os.path.splitext(nm)
            if (nm not in arymod_files) & (extn.lower() == '.pkl'):
                try:
                    ary = pkl.load(open(os.path.join(ary_sd, nm), 'rb'))
                    ary_ = replace_infreq_nts_in_ary(ary, notes_replacement)
                    pkl.dump(ary_, open(os.path.join(ary_mod_sd, nm), 'wb'))
                    if show_progress:
                        print(os.path.join(ary_sd, nm))
                except:
                    pass


"""
batch_replace_infreq_nts(
        notes_replacement, array_path='/Users/Wei/Desktop/midi_train/arry',
        array_mod_path='/Users/Wei/Desktop/midi_train/arry_modified', show_progress=True)
        
# indexing notes again!
notes_indexing(
        array_path='/Users/Wei/Desktop/midi_train/arry_modified', 
        tk_path='/Users/Wei/PycharmProjects/DataScience/Side_Project/Composer_transformer_gan/model/notes_dict_mod.pkl',
        print_num=True)
        
tk = pkl.load(open(tk_path, 'rb'))
len(json.loads(tk.get_config()['word_counts']))  # 15000
"""


class DataPreparation:

    @staticmethod
    def get_all_filenames(pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['']):
        filepath_list = [f.path for f in os.scandir(pths) if f.is_dir()]
        file_names = []
        for filepath, name_substr in itertools.product(filepath_list, name_substr_list):
            file_names += glob.glob(os.path.join(filepath, '*' + name_substr + '*.pkl'))
        print('Total number of files: {}'.format(len(file_names)))
        return file_names

    @staticmethod
    def cut_array(x, length, step, thres):
        # x: np 2d array: columns [notes_str, velocity, time_passed_since_last_note, notes_duration]
        result = []
        i = 0
        while length + i < x.shape[0]:
            result.append(x[i: (i + length)].tolist())
            i += step
        # not to waste the last part of music
        if length + i - x.shape[0] >= thres:
            result.append(x[x.shape[0]-length:].tolist())
        return result

    @staticmethod
    def batch_preprocessing(in_seq_len, out_seq_len, step=60,
                            pths='/Users/Wei/Desktop/midi_train/arry_modified', name_substr_list=['']):
        all_file_names = DataPreparation.get_all_filenames(pths=pths, name_substr_list=name_substr_list)
        length = in_seq_len + out_seq_len
        x_in, x_tar = [], []
        for filepath in all_file_names:
            ary = pkl.load(open(filepath, 'rb'))  # file.shape = (length, 2)
            if ary.shape[0] >= length:
                ary = DataPreparation.cut_array(ary, length, step, int(step/3))  # (n_samples, length, 2)
                ary = np.array(ary, dtype=object)
                x_in += ary[:, :in_seq_len, :].tolist()
                x_tar += ary[:, in_seq_len:, :].tolist()
        return np.array(x_in, dtype=object), np.array(x_tar, dtype=object)



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


