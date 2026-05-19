import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly  # for resampling
from pathlib import Path
import os

from file_functions import save_wav_file, save_csv_file, float_to_pcm_binary_string
from DFT_pipeline import iDFT_pipeline

def load_wav_as_generator_format(path, target_fs=None, mono=True, normalize=True):
    fs, data = wavfile.read(path)                 # fs: sample rate, data: ndarray
    # convert to float32 in [-1,1]
    if data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        # int16, int32, etc.
        maxv = float(np.iinfo(data.dtype).max)
        x = data.astype(np.float32) / maxv
    else:
        x = data.astype(np.float32)

    # stereo -> mono
    if mono and x.ndim == 2:
        x = x.mean(axis=1)

    # resample if requested
    if target_fs is not None and target_fs != fs:
        # use resample_poly for good quality
        x = resample_poly(x, target_fs, fs)
        fs = target_fs

    # optional normalize (avoid divide-by-zero)
    if normalize:
        m = np.max(np.abs(x))
        if m > 0:
            x = x / m

    return x.astype(np.float32), fs

def generate_key(length, fs, type, repeat_count = 1, silence_duration = 0.0):
    if type == 'chirp':
        t = np.linspace(0, length/fs, length)
        f0 = 20
        f1 = 1000
        key = np.cos(2*np.pi * (f0*t + (f1-f0)*t**2/(2*length/fs)))
    
    if repeat_count > 1 or silence_duration > 0:
        key = repeat_key_and_silence_pad(key, fs, repeat_count, silence_duration)

    return key.astype(np.float32)

def repeat_key_and_silence_pad(key, fs, repeat_count, silence_duration, silence_pad = True):
    
    # insert silence gaps between keys
    final_signal = np.empty(0, dtype=np.float32)
    if silence_pad:
        key_with_silence = np.concatenate((key, np.zeros(int(silence_duration*fs), dtype=np.float32)))
    else:
        key_with_silence = key
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

def main(params = '', test_signal_wav = None, test_signal = None):

    #user params
    if params == '':
        key_type = 'chirp'
        repeat_key_count = 1
        block_length = 1024
        cyclic_prefix_length = 32

        #length in samples
        length = 50000
        fs = 44100
    else:
        key_type = params['key_type']
        repeat_key_count = params['repeat_key_count']
        block_length = params['block_length']
        cyclic_prefix_length = params['cyclic_prefix_length']

        #length in samples, fs in Hz
        length = params['length']
        fs = params['fs']

    if test_signal_wav is None:
        test_signal = load_wav_as_generator_format('Week_1_Synchronisation_Aaron/Custom_test_01.wav', target_fs=fs)[0]
    else: #The test_signal is the wav supplied translated into appropriate format for the DFT stream.
        test_signal = load_wav_as_generator_format(test_signal_wav, target_fs=fs)[0]  # Normalize if int16

    key_time = length/fs
    print(f"Key duration: {key_time:.3f} seconds")

    #create desired key
      
    key = generate_key(length, fs, key_type, repeat_count = repeat_key_count, silence_duration = 0.1)
    #Set silence_pad to False to create a signal with no silence between keys
    #repeat_key = repeat_key_and_silence_pad(key, fs, repeat_count=repeat_key_count, silence_duration=0.1, silence_pad=False)
     
    print(len(test_signal))
    plt.plot(test_signal)
    plt.show()

    #save the signal as a wav file
    #print("saving....")
    #save_wav_file(test_signal, fs, key_type + '_' + str(repeat_key_count) + '_repeats.wav')
    #save_csv_file(test_signal, len(test_signal), key_type + '_' + str(repeat_key_count) + '_repeats.csv')
    
    print('Mapping to Gray code...')
    symbols_signal = gray_encode(float_to_pcm_binary_string(test_signal))
    print(f'Mapped {len(symbols_signal)} symbols.')
    
    x_cp_signal = iDFT_pipeline(symbols_signal, b_len=block_length, cp_len=cyclic_prefix_length)
    print(f'Generated OFDM signal with length {len(x_cp_signal)} samples.')

    #Get Key and iDFT x_cp as int16 for WAV file
    key_int16 = np.int16(key * 32767)
    tx_signal_int16 = np.int16(x_cp_signal * 32767)
    f_signal = np.concatenate((key_int16, tx_signal_int16))
    save_wav_file(f_signal, fs, key_type + '_' + str(repeat_key_count) + '_transmitted_signal.wav')

    

    '''mapped_signal_sample = mapped_signal[:100]  # Take the first 100 symbols for visualization
    plt.plot(mapped_signal.real, mapped_signal.imag, 'o')
    plt.title('Gray Code Mapping of Test Signal')
    plt.show()'''



if __name__ == "__main__":
    main()