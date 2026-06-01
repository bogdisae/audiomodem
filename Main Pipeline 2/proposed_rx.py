import numpy as np
from constellation import Constellation
from equaliser import Equaliser, GolayPairs
from proposed_synchroniser import Synchroniser, RepeatedChirpSync

class Rx:
    signal: np.ndarray
    correlation_dist: int
    cp_length: int
    block_length: int
    constellation: Constellation
    equaliser : Equaliser
    synchroniser : Synchroniser


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

    #AM additions
    pilot_spacing : int
    key_pilot_samples_spacing : int
    pilot_type : str
    pilot_config : str
    pair_count : int

    #SNS
    pilot_bins : np.ndarray
    a_history : list
    

    def __init__(self, constellation: Constellation, signal:np.ndarray, cp_length: int,
                 block_length: int, equaliser : Equaliser, synchroniser : Synchroniser, pilot_type, pilot_config,
                 early_samples = 30, pilot_spacing = 10, key_pilot_samples_spacing = 1024, f_low = 230, f_high = 14500):
        
        self.constellation = constellation
        self.signal = signal
        self.cp_length = cp_length
        self.block_length = block_length
        self.equaliser = equaliser
        self.synchroniser = synchroniser
        self.f_low = f_low
        self.f_high = f_high
        self.early_samples = early_samples


        # Calulate active subcarrier mask
        self.bin_low = int(np.ceil(f_low * block_length / synchroniser.fs))
        self.bin_high = int(np.floor(f_high * block_length / synchroniser.fs))
        self.active_bins = np.arange(self.bin_low, self.bin_high + 1)

        if pilot_config == "Block":
            self.key_pilot_samples_spacing = key_pilot_samples_spacing
            self.pilot_spacing = pilot_spacing   #No. blocks between pilot symbols - set to 0 for no repeats
            self.pilot_type = pilot_type
            self.pilot_config = pilot_config
            self.pair_count = 1 #Not supported yet for multi Golay pairs
        else:
            self.pilot_bins = self.active_bins[::pilot_spacing] #This is a simple way to select pilot bins for comb configuration - every nth active bin is a pilot
            self.data_bins = np.array([k for k in self.active_bins if k not in self.pilot_bins])

        self.a_history = [] 

    def decode_ofdm_block(self, block):

        #Going early here
        early_block_minus_cp = block[-(self.block_length + self.early_samples):-self.early_samples]
        #cp_discarded = block[-self.block_length:]
        Y = np.fft.fft(early_block_minus_cp)
        X = Y / self.H[0:len(Y)] # Zero-forcing

        # Phase correction for FFT window offset
        k = np.arange(len(X))
        phase_correction = np.exp(
            1j * 2 * np.pi * k * self.early_samples / self.block_length
        )

        X *= phase_correction

        '''APPLY SFO CORRECTION HERE'''
        if self.pilot_config == "Block":
            pass
        else: # Comb pilot configuration - simple linear interpolation of channel estimates across frequency
            pilots = X[self.pilot_bins]
            pilot_ref = self.constellation.default_pilot

            phase_error = np.angle(pilots / pilot_ref)

            f = self.pilot_bins * self.synchroniser.fs / self.block_length
            y = phase_error
            a_meas = np.sum(f* y) / np.sum(f**2)
            self.a_history.append(a_meas)
    
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

    def _decode_ofdm_region(self, start_index, end_index):
        self.ofdm_blocks = self.signal[start_index:end_index]
        self.extract_ofdm_blocks()
        return list(self.data_symbols)
    
    def decode(self):
        
        # Synchronise 
        key_start_index, self.synchronisation_index, second_peak_index = self.synchroniser.synchronise(self.signal, True)

        self.pilot_start_index = self.synchronisation_index + self.key_pilot_samples_spacing
        #print(f'Initial pilot start index: {self.pilot_start_index}')
        #print(f'key start index: {key_start_index}, synchronisation index: {self.synchronisation_index}')
        symbol_length = self.block_length + self.cp_length
        decoded_symbols = []  

        if self.pilot_config == "Block": #Block or comb
            if self.pilot_spacing == 0:
                current_pilot_start = self.pilot_start_index #CP length zeros transmitted after sync signal
                print("Estimating channel using pilot 1/1")
                self.H = self.equaliser.estimate(self.signal, current_pilot_start, self.pair_count,  True)

                data_start_index = current_pilot_start + self.equaliser.lengthInSamples
                decoded_symbols.extend(self._decode_ofdm_region(data_start_index, len(self.signal))) #No more pilots in signal
            else:
                current_pilot_start = self.pilot_start_index
                
                section_index = 0

                while current_pilot_start + self.equaliser.lengthInSamples <= len(self.signal):
                    
                    
                    self.H = self.equaliser.estimate(self.signal, current_pilot_start, self.pair_count,  False)

                    section_data_start = current_pilot_start + self.equaliser.lengthInSamples
                    section_data_end = min(
                        len(self.signal),
                        section_data_start + self.pilot_spacing * symbol_length,
                    )
                    #print(f"Estimating channel using pilot section {section_index}, section data start idx: {section_data_start}, end: {section_data_end}")
                    decoded_symbols.extend(self._decode_ofdm_region(section_data_start, section_data_end))

                    current_pilot_start = section_data_end
                    section_index += 1
        else: #pilot type is comb

            self.H = self.equaliser.estimate(self.signal, key_start_index, plot=True) #Assuming Chirp type estimator
            self.ofdm_blocks = self.signal[self.synchronisation_index+self.cp_length:]
            self.extract_ofdm_blocks()
            raise NotImplementedError("Comb pilot type not implemented yet")

        self.data_symbols = decoded_symbols
        self.decode_symbols()
        self.bits_to_bytes()


        


