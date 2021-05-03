import numpy as np

def get_flabel(frames, fnum, fps, seg_l):
    s_i = [int(seg_fn[0][:-4]) for seg_fn in frames]
    e_i = [s + fps * seg_l for s in s_i]
    e_i = map(round, e_i)
    e_i = map(int, e_i)

    label = np.zeros((fnum, 1))
    for s, e in zip(s_i, e_i):
        label[s:e] = 1
    return label