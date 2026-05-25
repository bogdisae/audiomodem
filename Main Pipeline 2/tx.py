import numpy as np
from scipy.signal import chirp
# import scipy.io.wavfile as wav
# import matplotlib.pyplot as plt
# from scipy.signal import convolve, correlate
# from scipy.io.wavfile import write
from constellation import Constellation
from equaliser import *

class Tx:
    data_bytes: np.ndarray
    constellation: Constellation
    cp_length: int
    block_length: int
    equaliser: Equaliser

    data_bits: np.ndarray
    data_symbols: np.ndarray
    ofdm_symbol_blocks: np.ndarray
    transmitted_signl: np.ndarray

    def __init__(self, constellation: Constellation, data_bytes: np.ndarray, equaliser : Equaliser, cp_length: int, block_length: int):
        self.constellation = constellation
        self.data_bytes = data_bytes
        self.equaliser = equaliser
        self.cp_length = cp_length
        self.block_length = block_length

    def chirp_signal(d = .3, f0 = 2, f1 =8000, savefile = False, fieldir="./bogdan/recordings" , fs=44100):
        t = np.linspace(0, d, int(fs * d), endpoint=False)
        signal = chirp(t, f0=f0, f1=f1, t1=d, method="linear")
        return signal

    # def white_noise(samples):
    #     return np.random.uniform(-1.0, 1.0, samples)

    def create_noise_key(noise_samples, n_taps):
        n = np.random.uniform(-1.0, 1.0, noise_samples)
        n[1::2] = 0.0
        key = np.zeros(2*noise_samples + n_taps)
        key[0:noise_samples] = n
        key[-noise_samples:] = n
        return n, key
    
    def bytes_to_bits(self):
        self.data_bits = np.unpackbits(self.data_bytes).astype(str)
    
    def encode_symbols(self):
        self.bytes_to_bits()
        self.data_symbols = self.constellation.bits_to_symbols(self.data_bits)

    def prep_ofdm_block(self, block):
        X = np.zeros(self.block_length, dtype=complex)

        half = len(block)
        X[1:half+1] = block
        # Hermitian symmetry for real signal
        X[-half:] = np.conj(block[::-1])
        ofdm_block = np.fft.ifft(X).real
        
        return np.concatenate([ofdm_block[-self.cp_length:], ofdm_block])

    def prep_ofdm_blocks(self):
        half_block_length = self.block_length // 2 -1
        padding_symbols = np.array(self.constellation.bits_to_symbols(('0','0'))) # symbol zeros
        pad_length = half_block_length - len(self.data_symbols) % half_block_length
        if pad_length > 0:
            padding = np.resize(padding_symbols, pad_length)
            self.data_symbols = np.concatenate([self.data_symbols, padding])
        blocks = self.data_symbols.reshape(-1, half_block_length)
        self.ofdm_symbol_blocks = [self.prep_ofdm_block(block) for block in blocks]
    
    def assemble_signal(self):
        ofdm_blocks = np.copy(self.ofdm_symbol_blocks)
        ofdm_blocks = ofdm_blocks / abs(max(ofdm_blocks))
        self.transmitted_signl = np.concatenate([self.equaliser.generate(), np.zeros(self.cp_length), np.concatenate(ofdm_blocks)])

    def encode(self):
        self.encode_symbols()
        self.prep_ofdm_blocks()
        self.assemble_signal()
    


