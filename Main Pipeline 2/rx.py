import numpy as np
from constellation import Constellation
from equaliser import Equaliser, RepeatedChirp

class Rx:
    signal: np.ndarray
    correlation_dist: int
    cp_length: int
    block_length: int
    constellation: Constellation
    equaliser : Equaliser

    synchronisation_index: int
    H: np.ndarray
    h: np.ndarray
    ofdm_blocks: np.ndarray
    data_symbols: np.ndarray
    data_bits: np.ndarray
    data_bytes: np.ndarray

    # SNS ADDITIONS xx
    f_low : int
    f_high : int
    bin_low : int
    bin_high : int
    active_bins : np.ndarray
    early_samples : int


    def __init__(self, constellation: Constellation, signal:np.ndarray, cp_length: int,
                 block_length: int, equaliser : Equaliser,
                 early_samples = 30, f_low = 230, f_high = 14500):
        
        self.constellation = constellation
        self.signal = signal
        self.cp_length = cp_length
        self.block_length = block_length
        self.equaliser = equaliser
        self.f_low = f_low
        self.f_high = f_high
        self.early_samples = early_samples

        # Calulate active subcarrier mask
        self.bin_low = int(np.ceil(f_low * block_length / equaliser.fs))
        self.bin_high = int(np.floor(f_high * block_length / equaliser.fs))
        # Using the equaliser fs feels messy but will do
        self.active_bins = np.arange(self.bin_low, self.bin_high + 1)


    def decode_ofdm_block(self, block):
        cp_discarded = block[-self.block_length:]
        Y = np.fft.fft(cp_discarded)
        X = Y / self.H[0:len(Y)] # Zero-forcing

        # Phase correction for FFT window offset
        k = np.arange(len(X))
        phase_correction = np.exp(
            1j * 2 * np.pi * k * self.early_samples / self.block_length
        )

        X *= phase_correction

        data_bins = X[self.active_bins]
        return data_bins

    def extract_ofdm_blocks(self):
        ofdm_symbol_length = self.block_length + self.cp_length
        pad_length = len(self.ofdm_blocks) % ofdm_symbol_length
        if pad_length > 0:
            self.ofdm_blocks = np.pad(self.ofdm_blocks, (0, ofdm_symbol_length - pad_length))
        self.ofdm_blocks = self.ofdm_blocks.reshape(-1, ofdm_symbol_length)

        self.data_symbols = []
        for block in self.ofdm_blocks:
            self.data_symbols.extend(self.decode_ofdm_block(block))

    def decode_symbols(self):
        self.data_bits = []
        self.data_bits = self.constellation.symbols_to_bits(self.data_symbols)

    def bits_to_bytes(self):
        self.data_bytes = np.packbits(np.array(self.data_bits).astype(np.uint8))
    
    def decode(self):
        
        # Synchronise 
        key_start_index, self.synchronisation_index = self.equaliser.synchronise(self.signal, True)
        self.H = self.equaliser.estimate(self.signal, key_start_index, True)

        # TRY GOING EARLY
        self.synchronisation_index = self.synchronisation_index + self.cp_length - self.early_samples

        # BOGDAN YOU FORGOT THIS LINE
        self.ofdm_blocks = self.signal[self.synchronisation_index:]

        self.extract_ofdm_blocks()
        self.decode_symbols()
        self.bits_to_bytes()


