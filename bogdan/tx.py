import numpy as np
from scipy.signal import chirp
# import scipy.io.wavfile as wav
# import matplotlib.pyplot as plt
# from scipy.signal import convolve, correlate
# from scipy.io.wavfile import write
from constellation import Constellation

class Tx:
    data_byte: np.ndarray
    constellation: Constellation

    def __init__(self, constellation: Constellation, data_bytes: np.ndarray):
        self.constellation = constellation
        data_bytes = data_bytes

    # def chirp_signal(d = .3, f0 = 2, f1 =8000,savefile = False, fieldir="./bogdan/recordings" , fs=44100):
    #     t = np.linspace(0, d, int(fs * d), endpoint=False)
    #     signal = chirp(t, f0=f0, f1=f1, t1=d, method="linear")
    #     return

    # def white_noise(samples):
    #     return np.random.uniform(-1.0, 1.0, samples)

    def create_noise_key(noise_samples, n_taps):
        n = np.random.uniform(-1.0, 1.0, noise_samples)
        n[1::2] = 0.0
        key = np.zeros(2*noise_samples + n_taps)
        key[0:noise_samples] = n
        key[-noise_samples:] = n
        return n, key
    


