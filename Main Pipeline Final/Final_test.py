print("Importing modules...")

from equaliser import Equaliser, RepeatedChirp, GolayPairs, WhiteNoise
from tx import Tx
from rx import Rx
from helper import pick_text_file, csv_to_data_bytes, pick_wav_file, normalise_signal, record_audio, plot_constellation_colour_seq, pick_csv_file, gen_colour_seq
from helper import csv_bytes_to_binary_sequence, calculate_ber, plot_constellation_colour
from file_decode_func import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import questionary
from scipy.io import wavfile
from scipy.io.wavfile import write
from constellation import Constellation

sampleRate = 48000

constellation = Constellation(2, {
    ('0', '0'): (1+1j)/np.sqrt(2),
    ('0', '1'): (-1+1j)/np.sqrt(2),
    ('1', '0'): (1-1j)/np.sqrt(2),
    ('1', '1'): (-1-1j)/np.sqrt(2)
}, {
    ('0', '0'): lambda s: (s.real >= 0) & (s.imag >= 0),
    ('0', '1'): lambda s: (s.real < 0) & (s.imag >=  0),
    ('1', '0'): lambda s: (s.real >=  0) & (s.imag < 0),
    ('1', '1'): lambda s: (s.real <  0) & (s.imag <  0),
})

def generate_final_sig(standard = True):
    transmitted_file = 