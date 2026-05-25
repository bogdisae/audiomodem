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
    transmitted_signal: np.ndarray


    # RECEIVER MUST MATCH THESE PARAMATERS!
    f_low : int
    f_high : int
    bin_low : int
    bin_high : int
    active_bins : np.ndarray

    def __init__(self, constellation: Constellation, data_bytes: np.ndarray, equaliser : Equaliser, cp_length: int, block_length: int,
                 f_low = 230, f_high = 14500):
        
        self.constellation = constellation
        self.data_bytes = data_bytes
        self.equaliser = equaliser
        self.cp_length = cp_length
        self.block_length = block_length

        # Calulate active subcarrier mask
        self.bin_low = int(np.ceil(f_low * block_length / equaliser.fs))
        self.bin_high = int(np.floor(f_high * block_length / equaliser.fs))
        # Using the equaliser fs feels messy but will do
        self.active_bins = np.arange(self.bin_low, self.bin_high + 1)


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

        # Positive-frequency active bins
        X[self.active_bins] = block

        # Hermitian symmetry
        X[-self.active_bins] = np.conj(block)

        ofdm_block = np.fft.ifft(X).real
        return np.concatenate([ofdm_block[-self.cp_length:], ofdm_block])

    def prep_ofdm_blocks(self):
        symbols_per_block = len(self.active_bins)

        padding_symbols = np.array(self.constellation.bits_to_symbols(('0', '0')))

        remainder = len(self.data_symbols) % symbols_per_block
        pad_length = symbols_per_block - remainder if remainder != 0 else 0

        if pad_length > 0:
            padding = np.resize(padding_symbols, pad_length)
            self.data_symbols = np.concatenate([self.data_symbols, padding])

        blocks = self.data_symbols.reshape(-1, symbols_per_block)

        self.ofdm_symbol_blocks = [
            self.prep_ofdm_block(block)
            for block in blocks
        ]
    
    def assemble_signal(self):
        ofdm_blocks = np.copy(self.ofdm_symbol_blocks)
        ofdm_blocks = ofdm_blocks / np.abs(np.max(ofdm_blocks))
        self.transmitted_signal = np.concatenate([self.equaliser.generate(), np.zeros(self.cp_length), np.concatenate(ofdm_blocks)])

    def encode(self):
        self.encode_symbols()
        self.prep_ofdm_blocks()
        self.assemble_signal()
    


