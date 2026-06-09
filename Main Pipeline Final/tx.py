import numpy as np
from scipy.signal import chirp
# import scipy.io.wavfile as wav
# import matplotlib.pyplot as plt
# from scipy.signal import convolve, correlate
# from scipy.io.wavfile import write
from constellation import Constellation
from equaliser import *
from ldpc import ldpc
from helper import plot_constellation

class Tx:
    header_filename: str
    data_bytes: np.ndarray
    constellation: Constellation
    cp_length: int
    block_length: int
    equaliser1: Equaliser
    equaliser2: Equaliser
    equalisere3: Equaliser

    data_bits: np.ndarray
    ldpc_bits: np.ndarray
    data_symbols: np.ndarray
    ofdm_symbol_blocks: np.ndarray
    transmitted_signal: np.ndarray


    # RECEIVER MUST MATCH THESE PARAMATERS!
    f_low : int
    f_high : int
    bin_low : int
    bin_high : int
    active_bins : np.ndarray
    use_ldpc: bool
    c: ldpc.code
    pilot_spacing : int
    key_pilot_samples_spacing : int


    def __init__(self, 
                 header_filename: str,
                 constellation: Constellation, 
                 data_bytes: np.ndarray, 
                 equaliser1 : Equaliser, 
                 equaliser2 : Equaliser, 
                 equaliser3 : Equaliser, 
                 cp_length: int, 
                 block_length: int, 
                 pilot_spacing: int, 
                 f_low: int, 
                 f_high: int,
                 fs: int = 48_000,
                 use_ldpc: bool = True):
        
        self.constellation = constellation
        self.data_bytes = data_bytes
        self.equaliser1 = equaliser1
        self.equaliser2 = equaliser2
        self.equaliser3 = equaliser3
        self.cp_length = cp_length
        self.block_length = block_length
        self.f_s = fs

        # Calulate active subcarrier mask
        self.bin_low = int(np.ceil(f_low * block_length / fs))
        self.bin_high = int(np.floor(f_high * block_length / fs))
        # Using the equaliser fs feels messy but will do
        self.active_bins = np.arange(self.bin_low, self.bin_high + 1)

        self.pilot_spacing = pilot_spacing
        
        self.c = ldpc.code('802.16', z=61)
        self.use_ldpc = use_ldpc

        self.header_filename = header_filename

    # def create_noise_key(noise_samples, n_taps):
    #     n = np.random.uniform(-1.0, 1.0, noise_samples)
    #     n[1::2] = 0.0
    #     key = np.zeros(2*noise_samples + n_taps)
    #     key[0:noise_samples] = n
    #     key[-noise_samples:] = n
    #     return n, key
    
    def assemble_header(self):
        # B: length of data
        data_length_bytes = len(self.data_bytes).to_bytes(4, byteorder='big')
        print(f"Data length: {len(self.data_bytes)} bytes")
        # C: filename
        filename_bytes = self.header_filename.encode('utf-8')
        print(f"Header filename: {self.header_filename}, length: {len(filename_bytes)} bytes")
        # A: length of entire header = len(A) + len(B) + len(C)
        header_length = 2 + 4 + len(filename_bytes)
        header_length_bytes = header_length.to_bytes(2, byteorder='big')

        # Complete header
        header_bytes = header_length_bytes + data_length_bytes + filename_bytes
        print(f"Header length: {header_length} bytes")
        header_array = np.frombuffer(header_bytes, dtype=np.uint8)
        self.data_bytes = np.concatenate([header_array, self.data_bytes])

    def bytes_to_bits(self):
        self.data_bits = np.unpackbits(self.data_bytes).astype(str)
    
    def encode_symbols(self):
        bits = self.ldpc_bits if self.use_ldpc else self.data_bits
        self.data_symbols = self.constellation.bits_to_symbols(bits)

    def ldpc(self):
        if self.use_ldpc:
            bits_padded = np.pad(self.data_bits, (0, self.c.K-len(self.data_bits)%self.c.K))
            ldpc_shaped = np.reshape(bits_padded, (-1, self.c.K))
            self.ldpc_bits = np.array([self.c.encode(ldpc_block) for ldpc_block in ldpc_shaped]).flatten().astype(str)

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

        if self.use_ldpc:
            # we map 35 ldpc blocks to 30 ofdm blocks
            if (symbols_per_block != 854): raise Exception("Standard requres 854 active bins")
            remainder = len(self.data_symbols) % (self.c.K*35)
            pad_length = 35*self.c.K - remainder if remainder != 0 else 0
        else:
            remainder = len(self.data_symbols) % symbols_per_block
            pad_length = symbols_per_block - remainder if remainder != 0 else 0

        if pad_length > 0:
            padding = np.resize(padding_symbols, pad_length)
            self.padded_data_symbols = np.concatenate([self.data_symbols, padding])
        else:
            self.padded_data_symbols = self.data_symbols.copy()

        if self.use_ldpc:
            ldpc_blocks = self.padded_data_symbols.reshape(-1, 35*self.c.K)
            interleaved_blocks = np.array([], dtype=complex)
            ldpc_skip_factor = 15839
            thirty_ofdm_block_length = 25620 # 30x854
            for ldpc_block in ldpc_blocks:
                interleaved_block = np.zeros(thirty_ofdm_block_length, dtype=complex)
                for i in range(len(ldpc_block)):
                    interleaved_block[(i*ldpc_skip_factor)%thirty_ofdm_block_length]= ldpc_block[i]
                interleaved_blocks=np.concatenate([interleaved_blocks ,interleaved_block])
            blocks = np.array(interleaved_blocks).reshape(-1, symbols_per_block)
            self.interleaved_blocks = blocks
        else:
            blocks = self.padded_data_symbols.reshape(-1, symbols_per_block)

        self.ofdm_symbol_blocks = [
            self.prep_ofdm_block(block)
            for block in blocks
        ]
    
    def assemble_signal(self):
        ofdm_blocks = np.copy(self.ofdm_symbol_blocks)
        ofdm_blocks = ofdm_blocks / np.abs(np.max(ofdm_blocks))
        
        
        chirp_seq = self.equaliser1.generate()
        Golay_seq = self.equaliser2.generate()
        pilot_symbol = self.equaliser3.generate()

        sections = []

        # 1000 samples of silence at the start - helps with clear transmission (first chirp cuts off)
        sections.append(np.zeros(1000))
        sections.append(chirp_seq)
        sections.append(Golay_seq)
        for i, block in enumerate(ofdm_blocks):
            # ADD IN THE -1 TO ADHERE TO STANDARD AKA 19 INSTEAD OF 20
            if i % (self.pilot_spacing - 1) == 0 and i != 0:
                sections.append(pilot_symbol)
                print(f'Pilot inserted before block {i}')
            sections.append(block)
        self.transmitted_signal = np.concatenate(sections)
    
    def encode(self):
        # self.assemble_header()
        self.bytes_to_bits()
        self.ldpc()
        self.encode_symbols()
        self.prep_ofdm_blocks()
        self.assemble_signal()


    


