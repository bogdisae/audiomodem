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

    data_bits: np.ndarray
    data_symbols: np.ndarray
    ofdm_symbol_blocks: np.ndarray

    def __init__(self, constellation: Constellation, data_bytes: np.ndarray, equaliser : Equaliser, cp_length: int, block_length: int):
        self.constellation = constellation
        self.data_bytes = data_bytes
        self.equaliser = equaliser
        self.cp_length = cp_length
        self.block_length = block_length

    
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

        return np.fft.irfft(X)

    def prep_ofdm_blocks(self):
        pad = self.block_length - len(self.data_symbols) % self.block_length
        padded = np.pad(self.data_symbols, (0, pad % self.block_length))  # % n avoids padding when already divisible
        blocks = padded.reshape(-1, self.block_length)
        self.ofdm_symbol_blocks = [self.prep_ofdm_block(block) for block in blocks]
    
    def assemble_signal(self):
        pass

    def encode(self):
        self.encode_symbols()
        self.prep_ofdm_blocks()
        self.assemble_signal()
    


