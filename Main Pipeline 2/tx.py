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
        ofdm_block = np.fft.ifft(X).real
        
        return np.concatenate([ofdm_block[-self.cp_length:], ofdm_block])

    def prep_ofdm_blocks(self):
        half_block_length = self.block_length // 2 -1
        pad = half_block_length - len(self.data_symbols) % half_block_length
        padded = np.pad(self.data_symbols, (0, pad % half_block_length))  # % n avoids padding when already divisible
        blocks = padded.reshape(-1, half_block_length)
        self.ofdm_symbol_blocks = [self.prep_ofdm_block(block) for block in blocks]
    
    def assemble_signal(self):
        ofdm_blocks = np.copy(self.ofdm_symbol_blocks)
        ofdm_blocks = ofdm_blocks / abs(max(ofdm_blocks))
        self.transmitted_signl = np.concatenate([self.equaliser.generate(), np.zeros(self.cp_length), np.concatenate(ofdm_blocks)])

    def encode(self):
        self.encode_symbols()
        self.prep_ofdm_blocks()
        self.assemble_signal()
    


