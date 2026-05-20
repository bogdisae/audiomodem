from transmit_functions import  save_wav_file
from scipy.io import wavfile
from scipy.signal import welch
from pathlib import Path
import numpy as np

def quick_plot_key_comparison(data_1, data_1_name, data_2, data_2_name, x_label='Sample Index', y_label='Amplitude', scale = 'linear'):
    import matplotlib.pyplot as plt

    if scale == 'log_y_and_phase':
        data_1 = 10 * np.log10(np.abs(data_1) + 1e-12)  # Add small value to avoid log(0)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data_1, color = 'blue')
    if scale == 'log_y_and_phase':
        plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.title(data_1_name)
    #plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(data_2, color='orange')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.title(data_2_name)
    #plt.legend()

    plt.tight_layout()
    plt.show()

def window_with_hann(signal):
    hann_window = np.hanning(len(signal))
    return signal * hann_window

def isolate_key_signal(recorded_signal_wav_path, sync_idx, params):
    
    start_idx = sync_idx
    end_idx = start_idx + params['length_of_key'] #Length of Chirp
    
    recorded_signal_wav = wavfile.read(recorded_signal_wav_path)[1] #Read wav file and get data array

    key_wav = recorded_signal_wav[start_idx:end_idx]

    print(f'Shape of KEY WAV: {len(key_wav)}')
    output_path = Path(recorded_signal_wav_path).parent / (params['key_type'] + '_isolated_key.wav')

    save_wav_file(key_wav, params['fs_record'], output_path)
    return output_path
    # Placeholder for isolating the chirp signal from the recorded audio
    
def estimate_channel_response(isolated_key, original_key, params):

    #blocks not needed for chirps - didn't go through the OFDM pipeline
    print(f'Beginning of Sent chirp: {original_key[:10]}')
    print(f'end of Sent chirp: {original_key[-10:]}')

    print(f'Beginning of Isolated chirp: {isolated_key[:10]}')
    print(f'end of Isolated chirp: {isolated_key[-10:]}')
    

    #Remove N?
    #rfft assumes real input and returns only non redundant positive frequencies, saving memory
    '''#ALIGN FFT bins with the BINS of the OFDM blocks'''
    #print(f'Length of isolated key: {len(isolated_key)}')
    #print(f'Length of original key: {len(original_key)}')
    #Take the DFT but aling frequency bins with OFDM blocks
    Y = np.fft.rfft(isolated_key, n=params['block_length'])
    S = np.fft.rfft(original_key, n=params['block_length'])
    '''Maybe TRY squared over squared'''
    H = Y[:params['block_length']//2-1] / S[:params['block_length']//2-1]  # Take only the bins corresponding to the OFDM subcarriers (excluding DC and Nyquist)

    #Corresponding frequencies of bins
    freqs = np.fft.rfftfreq(params['block_length'], d=1/params['fs'])


    quick_plot_key_comparison(isolated_key, 'Isolated Key', original_key, 'Original Key')
    quick_plot_key_comparison(np.abs(H), 'Estimated Channel Magnitude Response', np.angle(H), 'Estimated Channel Phase Response', x_label='Subcarrier Index', y_label='Magnitude / Phase (radians)')
    
    _, S_ss = welch(isolated_key, fs=params['fs'],
                window='hann', nperseg=params['block_length'], nfft=params['block_length'],
                return_onesided=True)
    _, S_yy = welch(original_key, fs=params['fs'],
                window='hann', nperseg=params['block_length'], nfft=params['block_length'],
                return_onesided=True)

    H_squared = S_ss / (S_yy + 1e-12)  # Add small value to avoid division by zero
    '''THIS FUNCTION NEEDS CHECKING - NOT SURE IF IT IS CORRECT'''
    quick_plot_key_comparison(H_squared, 'Estimated Channel Response (Squared)', np.angle(H_squared), 'Estimated Channel Phase Response (Squared)', x_label='Subcarrier Index', y_label='Magnitude / Phase (radians)', scale='log_y_and_phase')
    #print(f'Frequencies in bins: {freqs}')
    return H
