import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from pathlib import Path
import os

def generate_key(length, fs, type):
    if type == 'chirp':
        t = np.linspace(0, length/fs, length)
        f0 = 20
        f1 = 1000
        key = np.cos(2*np.pi * (f0*t + (f1-f0)*t**2/(2*length/fs)))
    
    return key.astype(np.float32)

def repeat_key_and_silence_pad(key, fs, repeat_count, silence_duration):
    # insert silence gaps between keys
    final_signal = np.empty(0, dtype=np.float32)
    key_with_silence = np.concatenate((key, np.zeros(int(silence_duration*fs), dtype=np.float32)))
    for i in range(repeat_count):
        final_signal = np.concatenate((final_signal, key_with_silence))

    '''#insert silence at the beginning of the signal to allow for real world operation time
    final_signal = np.insert(final_signal, 0, np.zeros(int(1*fs), dtype= np.float32))'''

    return final_signal

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

def gray_encode(binary_string):
    
    #create an empty complex array to hold the gray code symbols
    gray_length = len(binary_string) // 2
    gray_code = np.empty(gray_length, dtype=np.complex128)

    for i in range(gray_length):
        #most significant bit and least significant bit for the current symbol
        msb = int(binary_string[2*i])
        lsb = int(binary_string[2*i + 1])
        # Perform Gray code conversion (example implementation - adjust as needed)
        if msb == 0 and lsb == 0:
            gray_code[i] = complex(1, 1)
        elif msb == 0 and lsb == 1:
            gray_code[i] = complex(-1, 1)
        elif msb == 1 and lsb == 1:
            gray_code[i] = complex(-1, -1)
        else:  # msb == 1 and lsb == 0
            gray_code[i] = complex(1, -1)

    gray_code /= np.sqrt(2)  # Normalize to unit power
    return gray_code

def iDFT_pipeline(symbols, b_len = 1024, cp_len = 32):
    
    #Chop into OFDM symbols  
    blocks = [symbols[i:i+511] for i in range(0, len(symbols), 511)]  # Reshape into blocks of b_len symbols

    print(f'Shape of Blocks: {len(blocks)}')
    iDFT_output = []
    for block in blocks:
        
        if len(block) < 511:
            block = np.pad(block, (0, 511 - len(block)))  # pad last block
        # Place modulated complex symbols from 1 block into specific frequency bins - subcarriers
        # No. frequency bins is b_len. 
        X = np.zeros((b_len), dtype=np.complex128)

        X[1:512] = np.block([block[:511]])
        #Enfore Hermiticity for real time domain signal
        X[513:] = np.conj(X[1:512][::-1])

        x = np.fft.ifft(X).real
        #Should be real from Hermitian symmetry, but take real part to avoid numerical issues

        #apply cyclic prefix
        cp = x[-cp_len:]

        x_cp = np.concatenate((cp, x))

        x_cp /= np.max(np.abs(x_cp))  # Normalize to prevent clipping

        iDFT_output.append(x_cp)
    tx_signal = np.concatenate(iDFT_output)
    return tx_signal

def main():

    #user params
    key_type = 'chirp'
    repeat_key_count = 5
    block_length = 1024
    cyclic_prefix_length = 32

    #length in samples, fs in Hz
    length = 5000
    fs = 44100
    key_time = length/fs
    print(f"Key duration: {key_time:.3f} seconds")

    #create desired key
      
    key = generate_key(length, fs, key_type)
    
    #Assemble multiple keys together to create a longer signal with padding
    
    test_signal = repeat_key_and_silence_pad(key, fs, repeat_count=repeat_key_count, silence_duration=0.1)

    
    print(len(test_signal))
    plt.plot(test_signal)
    plt.show()

    #save the signal as a wav file
    print("saving....")
    save_wav_file(test_signal, fs, key_type + '_' + str(repeat_key_count) + '_repeats.wav')
    save_csv_file(test_signal, len(test_signal), key_type + '_' + str(repeat_key_count) + '_repeats.csv')
    
    print('Mapping to Gray code...')
    symbols_signal = gray_encode(float_to_pcm_binary_string(test_signal))
    print(f'Mapped {len(symbols_signal)} symbols.')
    
    x_cp_signal = iDFT_pipeline(symbols_signal, b_len=block_length, cp_len=cyclic_prefix_length)
    print(f'Generated OFDM signal with length {len(x_cp_signal)} samples.')

    tx_signal_int16 = np.int16(x_cp_signal * 32767)
    save_wav_file(tx_signal_int16, fs, key_type + '_' + str(repeat_key_count) + '_mapped_repeats.wav')

    

    '''mapped_signal_sample = mapped_signal[:100]  # Take the first 100 symbols for visualization
    plt.plot(mapped_signal.real, mapped_signal.imag, 'o')
    plt.title('Gray Code Mapping of Test Signal')
    plt.show()'''



if __name__ == "__main__":
    main()