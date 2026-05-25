import numpy as np
from constellation import Constellation

class Rx:
    signal: np.ndarray
    correlation_dist: int
    # n_taps: int
    cp_length: int
    block_length: int
    constellation: Constellation

    # noise_key: np.ndarray
    synchronisation_index: int
    # windowed: np.ndarray
    # correlation: np.ndarray
    H: np.ndarray
    h: np.ndarray
    ofdm_blocks: np.ndarray
    data_symbols: np.ndarray
    data_bits: np.ndarray
    data_bytes: np.ndarray

    def __init__(self, constellation: Constellation, signal:np.ndarray, cp_length: int, block_length: int):
        self.constellation = constellation
        self.signal = signal
        self.cp_length = cp_length
        self.block_length = block_length

    def decode_ofdm_block(self, block):
        cp_discarded = block[-self.block_length:]
        Y = np.fft.fft(cp_discarded)
        X = Y / self.H[0:len(Y)] # Zero-forcing
        data_bins = X[1:self.block_length//2]
        return data_bins

    def extract_ofdm_blocks(self):
        block_length = self.block_length + self.cp_length
        pad_length = len(self.ofdm_blocks) % block_length
        if pad_length > 0:
            self.ofdm_blocks = np.pad(self.ofdm_blocks, (0, block_length - pad_length))
        self.ofdm_blocks = self.ofdm_blocks.reshape(-1, block_length)

        self.data_symbols = []
        for block in self.ofdm_blocks:
            self.data_symbols.extend(self.decode_ofdm_block(block))

    def decode_symbols(self):
        self.data_bits = []
        self.data_bits = self.constellation.symbols_to_bits(self.data_symbols)

    def bits_to_bytes(self):
        self.data_bytes = np.packbits(np.array(self.data_bits).astype(np.uint8))
    
    def decode(self):
        # self.synchronise_noise_key()
        # self.channel_estimate()

        self.extract_ofdm_blocks()
        self.decode_symbols()
        self.bits_to_bytes()


