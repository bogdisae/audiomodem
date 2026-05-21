import numpy as np

def bits_to_qpsk(bit_list):

    'Converts a binary bitstring into QPSK symbols'

    bit_list = np.array(bit_list)

    # Make sure even length
    bit_list = bit_list[:len(bit_list) - (len(bit_list) % 2)]

    symbols = np.array([
        (1 + 1j) if (bit_list[i], bit_list[i+1]) == (0, 0) else
        (-1 + 1j) if (bit_list[i], bit_list[i+1]) == (0, 1) else
        (-1 - 1j) if (bit_list[i], bit_list[i+1]) == (1, 1) else
        (1 - 1j)
        for i in range(0, len(bit_list), 2)
    ], dtype=complex)

    # THIS USES THE GRAY ENCODING

    return symbols / np.sqrt(2)

def frame_symbols(symbols, frame_size):

    'Splits the symbols into the specified size'

    return [symbols[i:i+frame_size] for i in range(0, len(symbols), frame_size)]

def ofdm_modulate(block, n_fft=1024):
    X = np.zeros(n_fft, dtype=complex)

    half = len(block)
    X[1:half+1] = block[:half]

    # Hermitian symmetry for real signal
    X[-half:] = np.conj(X[1:half+1][::-1])

    return np.fft.ifft(X)


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