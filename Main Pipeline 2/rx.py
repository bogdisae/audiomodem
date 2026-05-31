import numpy as np
from constellation import Constellation
from equaliser import Equaliser, RepeatedChirp
import matplotlib.pyplot as plt

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

    f_low : int
    f_high : int
    bin_low : int
    bin_high : int
    active_bins : np.ndarray
    early_samples : int
    pilot_spacing : int
    
    # Debug
    a_history : list

    def __init__(self, constellation: Constellation, signal:np.ndarray, cp_length: int,
                 block_length: int, equaliser : Equaliser,
                 early_samples = 30, f_low = 4000, f_high = 13000, pilot_spacing = 10):
        
        self.constellation = constellation
        self.signal = signal
        self.cp_length = cp_length
        self.block_length = block_length
        self.equaliser = equaliser
        self.f_low = f_low
        self.f_high = f_high
        self.early_samples = early_samples
        self.pilot_spacing = pilot_spacing

        # tracking parameters (NEW)
        self.a = 0.0          # SFO slope estimate

        # Calulate active subcarrier mask
        self.bin_low = int(np.ceil(f_low * block_length / equaliser.fs))
        self.bin_high = int(np.floor(f_high * block_length / equaliser.fs))
        self.active_bins = np.arange(self.bin_low, self.bin_high + 1)

        # Pilot calculation identical to transmitter
        self.pilot_bins = self.active_bins[::pilot_spacing] 
        self.data_bins = np.array([k for k in self.active_bins if k not in self.pilot_bins])

        # Debug
        self.a_history = []

    def decode_ofdm_block(self, block, block_index):
        cp_discarded = block[-self.block_length:]
        Y = np.fft.fft(cp_discarded)
        X = Y / self.H[0:len(Y)]  # Zero-forcing

        # Phase correction for FFT window offset
        k = np.arange(len(X))
        phase_correction = np.exp(
            1j * 2 * np.pi * k * self.early_samples / self.block_length
        )
        X *= phase_correction

        # -----------------------------
        # PILOT-BASED TRACKINGa
        # -----------------------------

        
        pilots = X[self.pilot_bins]
        pilot_ref = self.constellation.default_pilot

        # phase error
        phase_error = np.angle(pilots / pilot_ref)

        f = self.pilot_bins * self.equaliser.fs / self.block_length
        y = phase_error
        a_meas = np.sum(f * y) / np.sum(f * f) # Basically linear regression but origin stays at 0
        self.a_history.append(a_meas)


        # # ---------------- DEBUG PLOT ----------------

        # # sort for clean plotting (VERY important for readability)
        # idx = np.argsort(f)
        # f_sorted = f[idx]
        # y_sorted = y[idx]

        # # slope (already computed)
        # a = a_meas

        # # fitted line
        # y_fit = a * f_sorted

        # plt.figure()

        # # measured data
        # plt.plot(f_sorted, y_sorted, 'o-', label="Measured phase error")

        # # fitted line
        # plt.plot(f_sorted, y_fit, '--', label=f"Fit: y = {a:.3e} * f")

        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Phase (radians)")
        # plt.title(f"Phase offset vs frequency (block {block_index})")

        # plt.grid(True)
        # plt.legend()

        # # show slope as annotation
        # plt.text(
        #     0.05, 0.95,
        #     f"Slope a = {a:.3e} rad/Hz",
        #     transform=plt.gca().transAxes,
        #     verticalalignment='top'
        # )

        # plt.show()

        # # --------------------------------------------

        # a_meas_per_block = a_meas_per_block / block_index
        # self.a = 

        # extract data
        data_bins = X[self.data_bins]

        return data_bins

    def extract_ofdm_blocks(self):
        ofdm_symbol_length = self.block_length + self.cp_length
        pad_length = len(self.ofdm_blocks) % ofdm_symbol_length
        if pad_length > 0:
            self.ofdm_blocks = np.pad(self.ofdm_blocks, (0, ofdm_symbol_length - pad_length))
        self.ofdm_blocks = self.ofdm_blocks.reshape(-1, ofdm_symbol_length)

        self.data_symbols = []
        for i, block in enumerate(self.ofdm_blocks):
            self.data_symbols.extend(self.decode_ofdm_block(block, i))

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
    
    def decode(self):
        
        # Synchronise 
        key_start_index, self.synchronisation_index = self.equaliser.synchronise(self.signal, True)
        self.H = self.equaliser.estimate(self.signal, key_start_index, True)

        #Diagnostic prints
        #print(f"NaN in H: {np.sum(np.isnan(self.H))}, Inf in H: {np.sum(np.isinf(self.H))}")
        #print(f"sync_index: {self.synchronisation_index}, signal length: {len(self.signal)}")

        # TRY GOING EARLY
        self.synchronisation_index = self.synchronisation_index + self.cp_length - self.early_samples

        # BOGDAN YOU FORGOT THIS LINE
        self.ofdm_blocks = self.signal[self.synchronisation_index:]

        self.extract_ofdm_blocks()
        self.decode_symbols()
        self.bits_to_bytes()


