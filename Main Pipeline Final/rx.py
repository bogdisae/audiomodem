import numpy as np
from constellation import Constellation
from equaliser import Equaliser, RepeatedChirp, GolayPairs
from helper import plot_complex_arrays_separate

class Rx:
    signal: np.ndarray
    correlation_dist: int
    cp_length: int
    block_length: int
    constellation: Constellation
    equalisers : list[Equaliser]
    sfoEqualiser: Equaliser

    H: np.ndarray
    h: np.ndarray
    ofdm_blocks: np.ndarray
    data_symbols: np.ndarray
    data_bits: np.ndarray
    data_bytes: np.ndarray

    f_low : int
    f_high : int
    bin_low : int
    bin_high : int
    active_bins : np.ndarray
    early_samples : int

    key_start_estimates : list # A list of when each equaliser key starts (predicted from sync logic)
    data_start_estimate : int  # Sync logic predicts when the data block begins

    def __init__(self, constellation: Constellation, signal:np.ndarray, cp_length: int,
                 block_length: int, equalisers : list[Equaliser], sfoEqualiser : Equaliser,
                 early_samples = 30, f_low = 2000, f_high = 12000):
        
        self.constellation = constellation
        self.signal = signal
        self.cp_length = cp_length
        self.block_length = block_length
        self.equalisers = equalisers
        self.sfoEqualiser = sfoEqualiser
        self.f_low = f_low
        self.f_high = f_high
        self.early_samples = early_samples

        # Calulate active subcarrier mask
        self.bin_low = int(np.ceil(f_low * block_length / equalisers[0].fs))
        self.bin_high = int(np.floor(f_high * block_length / equalisers[0].fs))
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

        TEMPOERARY_global_rotation = np.exp(1j * np.deg2rad(15))

        X *= phase_correction
        # X *= TEMPOERARY_global_rotation

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
        #Check for NaN/Inf in data symbols - give warning
        symbols = np.array(self.data_symbols)
        bad = ~np.isfinite(symbols)
        if bad.any():
            print(f"WARNING: {bad.sum()} NaN/Inf symbols at indices {np.where(bad)[0][:10]}")

        self.data_bits = []
        self.data_bits = self.constellation.symbols_to_bits(self.data_symbols)

    def bits_to_bytes(self):
        self.data_bytes = np.packbits(np.array(self.data_bits).astype(np.uint8))

    def sync_and_estimate(self):
        # Calculate offsests of each equaliser to the data
        offset = 0
        for equaliser in self.equalisers:
            equaliser.preambleStartOffset = offset
            offset += equaliser.lengthInSamples
        preambleTotalLength = offset
        
        preamble_start_estimates = [] # Stores where each synchroniser thinks the WHOLE preamble starts

        # Synchronise
        for equaliser in self.equalisers:
            if equaliser.sync:  
                local_start = equaliser.synchronise(self.signal, False)
                preamble_start_estimate = local_start - equaliser.preambleStartOffset
                preamble_start_estimates.append(preamble_start_estimate)

            else:
                preamble_start_estimates.append(None)

        # Logic to choose sync estimate (e.g just use the first sync estimate. Could break if None)
        preamble_start = preamble_start_estimates[0]
        self.data_start_estimate = preamble_start + preambleTotalLength
        # Account for cyclic prefix and going early. 
        decode_start = self.data_start_estimate + self.cp_length - self.early_samples

        # Use the sync estimate to find where you think EVERY equaliser starts
        self.key_start_estimates = []
        for equaliser in self.equalisers:
            self.key_start_estimates.append(preamble_start + equaliser.preambleStartOffset)

        # Estimate
        channel_estimates = [] # Will be a list of np.ndarray corresponding to each estimate
        for idx, equaliser in enumerate(self.equalisers):    
            if equaliser.est:
                key_start_index = self.key_start_estimates[idx]
                print("Key start index for golay", key_start_index)
                channel_estimates.append(equaliser.estimate(self.signal, key_start_index))
            else:
                channel_estimates.append(None)

        ## DEBUGGING:::

        plot_complex_arrays_separate(channel_estimates, ["Chirp", "Golay"])

        # Logic to choose which channel estimate to use (e.g just use the second. Could break if None)
        self.H = channel_estimates[1]

        # BOGDAN YOU FORGOT THIS LINE AGAIN FFS
        self.ofdm_blocks = self.signal[decode_start:]

    def initial_SFO_estimate(self):
        for idx, equaliser in enumerate(self.equalisers):
            if type(equaliser) is GolayPairs:
                key_start_idx = self.key_start_estimates[idx]
                equaliser.initial_SFO_estimate(self.signal, key_start_idx, False)
    
    def decode(self):

        self.sync_and_estimate()
        #self.initial_SFO_estimate()
        #self.SFO_correct()
        self.extract_ofdm_blocks()
        self.decode_symbols()
        self.bits_to_bytes()


