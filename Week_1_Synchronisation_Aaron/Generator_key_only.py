import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from pathlib import Path
import os

from file_functions import save_wav_file, save_csv_file, float_to_pcm_binary_string
from DFT_pipeline import iDFT_pipeline


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