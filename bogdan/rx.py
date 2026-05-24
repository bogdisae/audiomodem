import numpy as np
from helper import cross_correlation
from constellation import Constellation

class Rx:
    signal: np.ndarray
    correlation_dist: int
    n_taps: int
    cp_length: int
    block_length: int
    constellation: Constellation

    noise_key: np.ndarray
    synchronisation_index: int
    windowed: np.ndarray
    correlation: np.ndarray
    H: np.ndarray
    h: np.ndarray

    def __init__(self, constellation: Constellation, signal:np.ndarray, correlation_dist:int, n_taps:int, noise_key: np.ndarray, cp_length: int, block_length: int):
        self.constellation = constellation
        self.signal = signal
        self.correlation_dist = correlation_dist
        self.n_taps = n_taps
        self.noise_key = noise_key
        self.cp_length = cp_length
        self.block_length = block_length

    def detect_windowed_signal_peak(self):
        peak_width_estimate = self.n_taps // 5
        abs_windowed = abs(self.windowed)
        # First value that lies within the top peak_width_estimate
        threshold = np.sort(abs_windowed)[-peak_width_estimate]
        first_index = np.argmax(abs_windowed >= threshold)
        self.synchronisation_index = first_index - self.correlation_dist

    def synchronise_noise_key(self):
        self.correlation = cross_correlation(self.signal, self.correlation_dist)
        self.windowed = np.convolve(self.correlation, np.ones(self.correlation_dist))

        self.detect_windowed_signal_peak()

    def channel_estimate_division(self, received, known):
        '''Assuming synchronisation, use the known sent signal and the received signal to estimate impulse response'''
        fftlen = len(received)
        Y = np.fft.fft(received, n = fftlen)
        X = np.fft.fft(known, n = fftlen)
        self.H = Y/X
        self.h = np.fft.ifft(self.H)

    def channel_estimate(self):
        y = self.signal[self.synchronisation_index: self.synchronisation_index + self.correlation_dist]
        self.channel_estimate_division(y, self.noise_key)

    def extract_ofdm_block():
        pass

    def decode_symbols():
        pass

    def bits_to_bytes():
        pass
    
    def decode(self):
        self.synchronise_noise_key()
        self.channel_estimate()

        self.extract_ofdm_block()
        self.decode_symbols()
        self.bits_to_bytes()


