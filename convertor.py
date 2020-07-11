import numpy as np
import string
import sys
sys.path.append('/Users/Wei/Library/Python/2.7/lib/python/site-packages')
import mido


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
        max_len = max(tracks_len)
        min_n_msg = max_len * min_msg_pct
        all_arys = []
        for i in range(len(mid.tracks)):
            if len(mid.tracks[i]) > min_n_msg:
                ary_i = Conversion.track2seq(mid.tracks[i])
                ary_i += [[0]*88]*(max_len-len(ary_i))
                all_arys.append(ary_i)
        all_arys = np.array(all_arys)
        all_arys = all_arys.max(axis=0)
        sums = all_arys.sum(axis=1)
        # trim
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


"""
import matplotlib.pyplot as plt
import os

midifile_path = '/Users/Wei/Desktop/piano_classic/Beethoven/sonata_variations'
mid = mido. MidiFile(os.path.join(midifile_path, 'beethoven_variations_ich_denke_dein_(c)soltau.mid'), clip=True)

tmp = Conversion.mid2arry(mid)

tmp = tmp[:1000]
plt.plot(range(tmp.shape[0]), np.multiply(np.where(tmp > 0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
plt.show()

mid_new = Conversion.arry2mid(tmp, 545455)
mid_new.save('mid_new.mid')
"""












