#isolate the chirp signal
# window with Hann if needed
#FFT both chirps
# Divide R[k]/S[k]

from Generator_key_only import load_wav_as_generator_format
from file_functions import  save_wav_file
from scipy.io import wavfile
import numpy as np

def isolate_key_signal(recorded_signal_wav_path, sync_idx, params):
    
    start_idx = sync_idx
    end_idx = start_idx + params['length'] #Length of Chirp
    
    recorded_signal_wav = wavfile.read(recorded_signal_wav_path)[1] #Read wav file and get data array

    key_wav = recorded_signal_wav[start_idx:end_idx]

    print(f'Shape of KEY WAV: {len(key_wav)}')
    key_float32 = load_wav_as_generator_format(recorded_signal_wav_path, target_fs=params['fs_record'])[0][start_idx: end_idx]
    return key_float32
    #save_wav_file(key_wav, params['fs_record'], 'isolated_key.wav')

    # Placeholder for isolating the chirp signal from the recorded audio
    
def estimate_channel_response(isolated_key, original_key, params):

    #blocks not needed for chirps - didn't go through the OFDM pipeline
    print(f'Beginning of Sent chirp: {original_key[:10]}')
    print(f'end of Sent chirp: {original_key[-10:]}')

    print(f'Beginning of Isolated chirp: {isolated_key[:10]}')
    print(f'end of Isolated chirp: {isolated_key[-10:]}')
    '''IF VALUES ARE NOT NEAR 0, THEN HANN WINDOWING IS NEEDED'''
    '''IF THE ENDS AND BEGINNING DO NOT MATCH FROM ORIGINAL TO ISOLATED, THEN SYNCHRONISATION IS POOR'''

    Y = np.fft.fft(isolated_key, n=params['block_length'])
    S = np.fft.fft(original_key, n=params['block_length'])
    H = Y / S
    return H
