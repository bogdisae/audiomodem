import numpy as np
from scipy.signal import chirp
from pathlib import Path

def save_wav_file(data, fs, filename):
    """Save `data` to a WAV file next to this script as PCM16.

    This avoids hard-coded/incorrect directories and converts float arrays
    into signed 16-bit PCM appropriate for `scipy.io.wavfile.write`.
    """

    # Clean filename (handle Windows/Unix paths)
    f_name = Path(filename).name

    # Ensure extension
    if not f_name.endswith('.wav'):
        f_name += '.wav'

    output_name = f_name.replace('.wav', '_Output.wav')

    # Determine output path: same folder as this script
    out_path = Path(__file__).parent / output_name

    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare PCM data
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.floating):
        # Normalize if necessary then convert to int16
        max_abs = float(np.max(np.abs(arr))) if arr.size else 1.0
        if max_abs > 1.0:
            arr = arr / max_abs
        pcm = (arr * 32767).astype(np.int16)
    else:
        pcm = arr.astype(np.int16)

    write(str(out_path), fs, pcm)

    print(f"Saved WAV file: {out_path}")

def save_csv_file(samples, length, filename):
    csv_data = ','.join(str(b) for b in samples[:length])

    f_name = filename.split('/')[-1]
    print(f_name)
    
    # Determine output path: same folder as this script
    out_path = Path(__file__).parent / f_name

    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        f.write(csv_data)

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

    gap_len : int
        Number of zeros inserted between symbols

    Returns
    -------
    tx_signal : np.array
        Full transmit waveform
    """

    tx = []

    # Optional preamble
    if preamble is not None:
        tx.extend(preamble)

    for block in ofdm_blocks:

        # Add CP
        block_cp = add_cyclic_prefix(block, cp_len)
        tx.extend(block_cp)


    return np.array(tx)