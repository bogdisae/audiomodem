import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
import os

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

def float_to_pcm_binary_string(values, bits=8):
    
    max_int = 2**(bits - 1) - 1
    min_int = -2**(bits - 1)
    
    binary_string = ''

    for value in values:
        int_value = int(value * max_int)
        int_value = max(min_int, min(max_int, int_value))  # clip
    
        # Convert to two's complement binary
        binary_string += format(int_value & (2**bits - 1), f'0{bits}b')
    return binary_string
