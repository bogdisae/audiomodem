import numpy as np

class Constellation:
    bits_per_symbol: int
    constellation: dict
    def __init__(self, bits_per_symbol, constellation):
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation
    
    def bits_to_symbols(self, bits):
        print("Bits to symbols, bits length:", len(bits))
        if len(bits)%self.bits_per_symbol!=0:
            raise Exception(f"Bit string not divisible by {self.bits_per_symbol}")
            # Pad bits instead?
        group_bits = [tuple(bits[i:i+self.bits_per_symbol]) for i in range(0, len(bits), self.bits_per_symbol)]
        print(f'Grouped bits into {len(group_bits)} symbols')
        return np.array([self.constellation[b] for b in group_bits], dtype=complex) # could be made faster with numpy
    
    def symbols_to_bits(self, symbols):
        pass

    def symbols_to_colors(self, symbols):
        # Map symbols to colors for visualization
        symbol_colors = {
            (1+1j)/np.sqrt(2): 'blue',
            (-1+1j)/np.sqrt(2): 'orange',
            (1-1j)/np.sqrt(2): 'green',
            (-1-1j)/np.sqrt(2): 'red'
        }
        colour_list = [symbol_colors[s] for s in symbols]
        print(f'Mapped constellation symbols to a list of {len(colour_list)} colors')
        return colour_list


# def bits_to_qpsk(bit_list, constellation:Constellation): Replaced by Constellation class

#     'Converts a binary bitstring into QPSK symbols'

#     bit_list = np.array(bit_list)

#     symbols = constellation.bits_to_symbols(bit_list)

#     # THIS USES THE GRAY ENCODING

#     return symbols / np.sqrt(2)

def frame_symbols(symbols, frame_size):
    'Splits the symbols into the specified size'
    framed_symbols = [symbols[i:i+frame_size] for i in range(0, len(symbols), frame_size)]

    if len(framed_symbols[-1]) < frame_size:
        print(f"Warning: Last frame has only {len(framed_symbols[-1])} symbols, expected {frame_size}. Consider padding.")
        framed_symbols[-1] = np.pad(framed_symbols[-1].astype(complex), (0, frame_size - len(framed_symbols[-1])), mode='constant', constant_values=(1+1j)/np.sqrt(2))
        print(f"Last frame after padding: {len(framed_symbols[-1])} symbols")

    return framed_symbols

def ofdm_modulate(block, n_fft=1024):
    X = np.zeros(n_fft, dtype=complex)

    half = len(block)
    X[1:half+1] = block
    # Hermitian symmetry for real signal
    X[-half:] = np.conj(block[::-1])
    assert np.max(np.abs(np.imag(block))) < pow(1.0, -15), \
    "Complex values detected in OFDM blocks"

    return np.fft.ifft(X).real # floating point imginary compoentn


def add_cyclic_prefix(signal, cp_len):
    'Adds cyclic prefix to one OFDM symbol'
    prefix = signal[-cp_len:]
    return np.concatenate([prefix, signal])

def bytes_csv_to_bits(text):

    'Convert a comma-separated string of byte values into a flat list of bits.'

    byte_list = [int(x.strip()) for x in text.split(",") if x.strip() != ""]

    bit_list = []
    for byte in byte_list:
        bits = format(byte, "08b")  # 8-bit binary string
        bit_list.extend(int(b) for b in bits)

    return bit_list

def text_to_bits(text):
    #Convert text to utfa-8 bytes, then to bits
    byte_array = text.encode('utf-8')
    return [int(b) for byte in byte_array for b in format(byte, '08b')]

def build_transmit_signal(ofdm_blocks,
                          cp_len=128,
                          preamble=None):
    """
    Builds one long transmit waveform.

    Parameters
    ----------
    ofdm_blocks : list of np.arrays
        Time-domain OFDM symbols

    cp_len : int
        Cyclic prefix length

    preamble : np.array or None
        Optional chirp/preamble at start

    Returns
    -------
    tx_signal : np.array
        Full transmit waveform
    """

    tx = []

    # Add small initial silence (just trust me its useful)
    silence_len = 1000
    tx.extend(np.zeros(silence_len))

    # Optional preamble
    if preamble is not None:
        tx.extend(preamble)

    for block in ofdm_blocks:

        # Add CP
        block_cp = add_cyclic_prefix(block, cp_len)
        tx.extend(block_cp)


    return np.array(tx)