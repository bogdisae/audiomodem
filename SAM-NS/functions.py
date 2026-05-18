import numpy as np
from scipy.signal import chirp


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


def generate_chirp(fs, T, f0=100, f1=8000):
    t = np.linspace(0, T, int(fs*T), endpoint=False)
    signal = chirp(t, f0=f0, f1=f1, t1=T, method='linear')
    return signal / np.max(np.abs(signal))


def ofdm_modulate(block, n_fft=1024):
    X = np.zeros(n_fft, dtype=complex)

    half = len(block)
    X[1:half+1] = block[:half]

    # Hermitian symmetry for real signal
    X[-half:] = np.conj(X[1:half+1][::-1])

    return np.fft.ifft(X)

def bytes_csv_to_bits(text):

    'Convert a comma-separated string of byte values into a flat list of bits.'

    byte_list = [int(x.strip()) for x in text.split(",") if x.strip() != ""]

    bit_list = []
    for byte in byte_list:
        bits = format(byte, "08b")  # 8-bit binary string
        bit_list.extend(int(b) for b in bits)

    return bit_list