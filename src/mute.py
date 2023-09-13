'''
Code taken from repo for 'Music Translation: Generating Piano Arrangements in Different Playing Levels' by Matan Gover and Oded Zewi (ISMIR 2022)
'''


import numpy as np
import muspy

def to_pitchclass_pianoroll(pianoroll: np.ndarray):
    pc = np.zeros((pianoroll.shape[0], 12), bool)
    for c in range(12):
        pitches = range(c, 128, 12)
        pc[:, c] = np.logical_or.reduce(pianoroll[:, pitches], 1)
    return pc

def music_to_pianoroll(music: muspy.Music):
    return muspy.to_pianoroll_representation(music, encode_velocity=False)
