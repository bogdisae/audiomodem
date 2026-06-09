import numpy as np
from constellation import Constellation
from equaliser import Equaliser, RepeatedChirp, GolayPairs
from helper import plot_complex_arrays_separate
from ldpc import ldpc
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

class Rx:
    signal: np.ndarray
    cp_length: int
    block_length: int
    constellation: Constellation
    equalisers : list[Equaliser]
    sfoEqualiser: Equaliser
    early_samples : int
    fs : int

    H: np.ndarray
    h: np.ndarray
    ofdm_blocks: np.ndarray
    data_symbols: np.ndarray
    ldpc_bits: np.ndarray
    data_bits: np.ndarray
    data_bytes: np.ndarray

    # Variables dictating the frequencies that carry data
    f_low : int
    f_high : int
    bin_low : int
    bin_high : int
    active_bins : np.ndarray

    # Channel synchronisation, estimation and SFO estimation variables
    sfo_samples_per_second : float
    sfo_rad_per_index_per_block : float
    preamble_start_estimates : list[int] # Stores where each synchroniser thinks the WHOLE preamble starts
    preamble_start_estimate : int    # Overall estimate of preamble start, based on selection logic    
    preamble_total_length : int

    # Fuck ldpc
    use_ldpc: bool
    c: ldpc.code

    key_start_estimates : list[int]  # A list of when each equaliser key starts (predicted from sync logic)
    data_start_estimate : int        # Sync logic predicts when the data block begins
    decode_start : int               # This accounts for first data cyclic prefix and going early

    #Header handling
    header_length: int
    data_length: int
    filename: str
    payload: bytes

    def __init__(self, 
                 constellation: Constellation, 
                 signal:np.ndarray, 
                 cp_length: int,
                 block_length: int, 
                 equalisers : list[Equaliser], 
                 sfoEqualiser : Equaliser,
                 early_samples = 200, 
                 f_low = 2000,
                 f_high = 12000, 
                 fs: int = 48_000, 
                 use_ldpc: bool = True):
        
        self.constellation = constellation
        self.signal = signal
        self.cp_length = cp_length
        self.block_length = block_length
        self.equalisers = equalisers
        self.sfoEqualiser = sfoEqualiser
        self.f_low = f_low
        self.f_high = f_high
        self.fs = fs
        self.early_samples = early_samples


        # Calulate active subcarrier mask using receiver sample rate
        self.bin_low = int(np.ceil(f_low * block_length / self.fs))
        self.bin_high = int(np.floor(f_high * block_length / self.fs))
        self.active_bins = np.arange(self.bin_low, self.bin_high + 1)

        self.c = ldpc.code('802.16', z=61)
        self.use_ldpc = use_ldpc
        self.preamble_start_estimates = []
        self.key_start_estimates = []

    def decode_ofdm_block(self, block, block_index):
        cp_discarded = block[-self.block_length:]
        Y = np.fft.fft(cp_discarded)
        X = Y / self.H[0:len(Y)] # Zero-forcing

        k = np.arange(len(X))
        # Phase correction for FFT window offset
        phase_correction = np.exp(
            1j * 2 * np.pi * k * self.early_samples / self.block_length
        )
        X *= phase_correction

        # Test paramaters
        # self.sfo_rad_per_index_per_block = 1.4e-4
        # self.sfo_rad_per_index_per_block = 0

        # Following line is neccesary to change constant for block length 4096 -> 6144 to inc cycprefix
        rad_per_idx_per_cycblock = self.sfo_rad_per_index_per_block * 1.5 

        # Phase correction for elapsed time since sync (where block length is 4096+2048 = 6144)
        CORRECTION = 0 # Experiment with this for SNS reasons
        blocks_since_sync = (self.preamble_total_length - CORRECTION) / 6144
        time_correction = np.exp(-1j * rad_per_idx_per_cycblock * k * blocks_since_sync)

        block_index_with_noise_accounted_for = block_index + block_index // 20
        sfo_correction = np.exp(-1j * rad_per_idx_per_cycblock * k * block_index_with_noise_accounted_for)
        
        X *= time_correction
        X *= sfo_correction

        data_bins = X[self.active_bins]
        return data_bins

    def extract_ofdm_blocks(self):
        ofdm_symbol_length = self.block_length + self.cp_length
        
        if self.use_ldpc:
            remainder = len(self.ofdm_blocks) % (ofdm_symbol_length*30)
            self.ofdm_blocks = self.ofdm_blocks[:-remainder] if remainder != 0 else self.ofdm_blocks
        else:
            remainder = len(self.ofdm_blocks) % ofdm_symbol_length
            pad_length = ofdm_symbol_length - remainder if remainder != 0 else 0
            padding_symbols = np.array(self.constellation.bits_to_symbols(('0', '0')))
   
            if pad_length > 0:
                padding = np.resize(padding_symbols, pad_length)
                self.ofdm_blocks = np.concatenate([self.ofdm_blocks, padding])
        
        self.ofdm_blocks_reshaped = self.ofdm_blocks.reshape(-1, ofdm_symbol_length)

        decoded_symbols = []
        for idx, block in enumerate(self.ofdm_blocks_reshaped):
            decoded_symbols.extend(self.decode_ofdm_block(block, idx))
        self.decoded_symbols = decoded_symbols

        if self.use_ldpc:
            deinterleaved_symbols = []
            thirty_ofdm_block_length = 25620 # 30x854
            ldpc_skip_factor = 15839
            grouped_ofdm_blocks = np.array(decoded_symbols).reshape(-1, thirty_ofdm_block_length)
            for interleaved_block in grouped_ofdm_blocks:
                ldpc_block = np.zeros(len(interleaved_block), dtype=complex)
                for i in range(len(interleaved_block)):
                    ldpc_block[i]=interleaved_block[(i*ldpc_skip_factor)%thirty_ofdm_block_length]
                deinterleaved_symbols.extend(ldpc_block)
            self.data_symbols = deinterleaved_symbols
        else:
            self.data_symbols = self.decoded_symbols


    def decode_symbols(self):
        #Check for NaN/Inf in data symbols - give warning
        symbols = np.array(self.data_symbols)
        bad = ~np.isfinite(symbols)
        if bad.any():
            print(f"WARNING: {bad.sum()} NaN/Inf symbols at indices {np.where(bad)[0][:10]}")

        if self.use_ldpc:
            self.ldpc_bits = self.constellation.symbols_to_bits(self.data_symbols)
        else:
            self.data_bits = self.constellation.symbols_to_bits(self.data_symbols)

    def ldpc(self):
        if self.use_ldpc:
            ldpc_shaped = np.array(self.ldpc_bits).reshape(-1, self.c.K*self.constellation.bits_per_symbol).astype(int)
            lut = np.array([25, -25])
            ldpc_shaped_weighted = lut[ldpc_shaped]
            # LDPC decoder returns LLRs for all N bits per block (information + parity)
            # We only want the K information bits per block
            info_bits_only = []
            for ldpc_block in ldpc_shaped_weighted:
                llrs_per_block, _ = self.c.decode(ldpc_block)
                # Extract only the first K bits (information bits) from the N-bit codeword
                info_bits_only.extend(llrs_per_block[:self.c.K])
            llrs = np.array(info_bits_only)
            self.data_bits = np.array(['0' if llr > 0 else '1' for llr in llrs])


    def bits_to_bytes(self):
        self.data_bytes = np.packbits(np.array(self.data_bits).astype(np.uint8))

    def sync(self):
        # Calculate offsets of each equaliser relative to the preamble start
        offset = 0
        for equaliser in self.equalisers:
            equaliser.preambleStartOffset = offset
            offset += equaliser.lengthInSamples
        self.preamble_total_length = offset

        for equaliser in self.equalisers:
            if equaliser.sync:
                local_start = equaliser.synchronise(self.signal, True)
                local_preamble_start_estimate = local_start - equaliser.preambleStartOffset
                self.preamble_start_estimates.append(local_preamble_start_estimate)
            else:
                self.preamble_start_estimates.append(None)

        # Logic to choose sync estimate (e.g. use the first sync estimate)
        self.preamble_start_estimate = self.preamble_start_estimates[0]
        self.data_start_estimate = self.preamble_start_estimate + self.preamble_total_length
        self.decode_start = self.data_start_estimate - self.early_samples

        self.key_start_estimates = []
        for equaliser in self.equalisers:
            self.key_start_estimates.append(self.preamble_start_estimate + equaliser.preambleStartOffset)

        # Let initial odfm_blocks array consist of the whole signal after decode start
        self.ofdm_blocks = self.signal[self.decode_start:]


    def estimate(self):
        channel_estimates = []  # Will be a list of np.ndarray corresponding to each estimate
        for idx, equaliser in enumerate(self.equalisers):
            if equaliser.est:
                key_start_index = self.key_start_estimates[idx]
                # (Note RepeatedChirp also plots its estimate in the function)
                channel_estimates.append(equaliser.estimate(self.signal, key_start_index, False))
            else:
                channel_estimates.append(None)

        # Logic to choose which channel estimate to use (e.g just use the second. Could break if None)
        # Use the repeated chirp estimate for now
        
        self.H = channel_estimates[1]



    def initial_SFO_estimate(self):
        slope = 0 # Slope represents the phase offset PER carrier index PER block

        for idx, equaliser in enumerate(self.equalisers):
            key_start_idx = self.key_start_estimates[idx]

            if type(equaliser) is GolayPairs:
                self.sfo_rad_per_index_per_block = equaliser.initial_SFO_estimate(self.signal, key_start_idx, self.bin_low, self.bin_high, False)
                pass

            if type(equaliser) is RepeatedChirp:
                #print("Using chirps to estimate SFO...")
                #self.sfo_rad_per_index_per_block = equaliser.initial_SFO_estimate(self.signal, key_start_idx, self.bin_low, self.bin_high, True)
                pass

        # To convert to sample drift / sec, consider the largest carrier 4096 (48000 Hz)
        # Multiplying the slope by 4096 leaves phase offset per block for 48000
        # At 48000 hz, if SFO = 1 sample/sec, this will correspond to a phase offset of 4096/48000 per block at the 4096 index
        # Therefore, SFO (in sample drift / sec) = slope * 4096 * 48000/4096 = slope * 48000

        self.sfo_samples_per_second = self.sfo_rad_per_index_per_block * 48000
        print("SFO in rad per carrier per block: ", self.sfo_rad_per_index_per_block)
        print("SFO in samples per second: ", self.sfo_samples_per_second)


    def separate_noise_symbols(self):
        symbol_length = self.block_length + self.cp_length

        self.noise_symbols = []
        data_chunks = []

        pos = 0

        while pos < len(self.ofdm_blocks):

            # 19 data symbols - BAD CODING PRACTICE I FUCKING KNOW
            data_end = pos + 19 * symbol_length
            data_chunks.append(self.ofdm_blocks[pos:data_end])

            # noise symbol
            noise_start = data_end
            # HERE WE ARE ASSUMING THE NOISE HAS LENGTH 4096 WITH 2048 CYCLIC PREFIX
            noise_end = noise_start + symbol_length
            if noise_start < len(self.ofdm_blocks):
                self.noise_symbols.append(
                    self.ofdm_blocks[noise_start:noise_end]
                )

            pos += 20 * symbol_length  # ASSUMES CYCLIC PREFIX FOR NOISE

        self.ofdm_blocks = np.concatenate(data_chunks)

    def pop_symbols_from_filename(self):
        self.filename = ''.join(c for c in self.filename if c.isprintable())


    def extract_header(self):
        # A: header length (2 bytes)
        self.header_length = int.from_bytes(self.data_bytes[:2], byteorder='big')
        print(f"Extracted header length: {self.header_length} bytes")

        # B: data length (4 bytes)
        self.data_length = int.from_bytes(self.data_bytes[2:6], byteorder='big')
        print(f"Extracted data length: {self.data_length} bytes (Not including header)")
        print(f'Need to extract the following No. Symbols: {(self.header_length + self.data_length) * 8 / self.constellation.bits_per_symbol} symbols after decode start')
        

        # C: filename (remaining bytes of header)
        filename_bytes = self.data_bytes[6:self.header_length]
        #print(f"Raw filename bytes: {list(filename_bytes)[:100]}")
        #print(f"As hex: {bytes(filename_bytes).hex()[:100]}")
        #print(f"Lossy decode: {bytes(filename_bytes).decode('utf-8', errors='replace')[:100]}")
        
        self.filename = bytes(filename_bytes).decode('utf-8')
        print(f'Uncleaned filename from header: {self.filename}')


        #Deal with corrupting symbols in filename - more rigerious
        self.pop_symbols_from_filename()
        # Extract payload
        self.payload = self.data_bytes[self.header_length:self.header_length + self.data_length]

        print(f"Extracted header - filename: {self.filename}, header length: {self.header_length}, data length: {self.data_length} bytes")


    def SFO_pilot_estimate(self, plot=True):

        fit_bins = 1706 # Use the first 20kHz of white noise
        
        pair_results = []

        # Search over 4x the estimated slope (accounting for the 20 blocks and cyclic prefix 1.5 factor. Shit code ik)
        initial_slope = self.sfo_rad_per_index_per_block * 20 * 1.5
        search_width = 20 * initial_slope

        # Set up trial slopes
        num_tries = 5000
        slopes = np.linspace(-search_width, search_width, num_tries)

        print("Length of individual noise symbol:", len(self.noise_symbols[0]))

        for pair_idx in range(len(self.noise_symbols)-2):

            Y0 = np.fft.fft(self.noise_symbols[pair_idx][2048:], n = 4096)
            Y1 = np.fft.fft(self.noise_symbols[pair_idx+1][2048:], n = 4096)

            R = (Y1 * np.conj(Y0))[:fit_bins]
            mag = np.abs(R)

            # Use only the most reliable X% of carriers. Can experiment with this
            threshold = np.percentile(mag, 0)
            mask = mag > threshold
            R = R[mask]
            k = np.arange(fit_bins)[mask]

            scores = []

            for slope in slopes:

                # Try rotating all points by the slope and see if they line up well
                corrected = (R * np.exp(-1j * slope * k))
                score = np.abs(np.sum(corrected / np.abs(corrected)))
                scores.append(score)

            scores = np.array(scores)
            best_idx = np.argmax(scores)
            slope = slopes[best_idx]

            pair_results.append(slope)

            if plot:

                corrected = (R * np.exp(-1j * slope * k))

                fig, ax = plt.subplots(1, 2, figsize=(14, 6))

                ax[0].plot(slopes, scores, marker='.', markersize=3)

                ax[0].plot(slope, scores[best_idx], 'ro', markersize=10)
                ax[0].axvline(slope, color="r", label="Estimated")

                initial_idx = np.argmin(np.abs(slopes - initial_slope))

                ax[0].plot(
                    initial_slope,
                    scores[initial_idx],
                    'go',
                    markersize=10
                )

                ax[0].axvline(
                    initial_slope,
                    color="g",
                    linestyle="--",
                    label="Initial"
                )

                ax[0].legend()
                ax[0].set_title(f"Alignment score pair {pair_idx}")

                ax[1].scatter(corrected.real, corrected.imag, s=3)
                ax[1].set_title("Corrected vectors")
                ax[1].axis("equal")

        if plot:
            plt.show()

        latest_slope = np.median(pair_results)
        calc = latest_slope / (20 * 1.5)

        samp_per_sec = (calc * 48000)

        print(f"Updating SFO "
            f"{self.sfo_rad_per_index_per_block}"
            f" -> {calc}")

        print(f"Updating SPS "
            f"{self.sfo_samples_per_second}"
            f" -> {samp_per_sec}")

        self.sfo_rad_per_index_per_block = calc
        self.sfo_samples_per_second = samp_per_sec
        
    def decode(self):

        self.sync()
        self.estimate()
        self.separate_noise_symbols()
        self.initial_SFO_estimate()
        self.SFO_pilot_estimate()
        self.extract_ofdm_blocks()
        self.decode_symbols()
        self.ldpc()
        self.bits_to_bytes()
        #self.extract_header()

        print("Final bit at index:", len(self.data_bits))
