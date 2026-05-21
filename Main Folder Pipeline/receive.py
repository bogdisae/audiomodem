#import relevant libraries
from pathlib import Path
import questionary
from scipy.io import wavfile
from receive_functions import normalise_signal, key_matched_filter, record_audio, generate_key
#from transmit_functions import save_wav_file
'''
MERGE CONFLICT REDUNDANT
from receive_functions import normalise_signal, chirp_matched_filter, record_audio, generate_chirp'''
#from transmit_functions import save_wav_file
import numpy as np
import matplotlib.pyplot as plt

def pick_wav_file(prompt_text: str, folder: Path) -> str:
    wav_files = sorted(folder.glob('*.wav'))
    if not wav_files:
        raise FileNotFoundError(f'No .wav files found in {folder}')

    choice = questionary.select(
        prompt_text,
        choices=[path.name for path in wav_files],
    ).ask()
    if choice is None:
        raise SystemExit('No file selected')
    return str(folder / choice)

def main(params):

    # Choose how to access the audio file

    mode = questionary.select("Do you want to record audio or select an existing file?",
        choices=["Record audio", "Select file"]
    ).ask()
    if mode is None: raise SystemExit("No option selected")
    if mode == "Select file":
        selected_path = pick_wav_file("Select a WAV file:", Path("./Main Folder Pipeline/Audio Files"))
    elif mode == "Record audio":
        print("Recording mode selected")
        # TO DO: CALL A RECORDING FUNCTION, OR SOMETHING 
        record_audio(params['record_duration'], params['fs_record'], filename=params['recording_name'])
        selected_path = Path(__file__).parent / 'Audio Files'/ params['recording_name']
    # Load the audio file
    fs_rx, rxSig = wavfile.read(selected_path)
    rxSig = normalise_signal(rxSig)

    # AT THIS POINT (HOPEFULLY) WE ARE SYNCHRONISED
    # NOW LETS FIND THE CHANNEL RESPONSE

    try:
        sync_index = key_matched_filter(rxSig, params['fs_record'], params['length_of_key'] / params['fs_record'], params['f0'], params['f1'], params['key_type'])
    except ValueError as e:
        print("Error during matched filtering:", e)
        raise

    print(f"{params['key_type']} starts at sample:", sync_index)

    end_idx = sync_index + params['length_of_key'] #Length of Chirp
    isolated_key = rxSig[sync_index:end_idx]
    Y = np.fft.rfft(isolated_key, n=params['block_length']) # Why do we use the block length as the DFT length? - Sam


    '''
    MERGE CONFLICT REDUNDANT
    # Code below is messy - will need to generalise for repeated chirps / other keys
    if params['key_type'] == 'chirp' and params['repeat_key_count'] == 1:
        chirp = generate_chirp(params['fs_record'], params['key_length'], 100, 8000)''' 
    
    key = generate_key(params['fs_record'], params['length_of_key'] / params['fs_record'], params['f0'], params['f1'], params['key_type'])
    S = np.fft.rfft(key, n=params['block_length'])

    eps = 1e-12  # Prevent divide-by-zero instability
    H = Y[1:-1] / (S[1:-1] + eps) # Remove DC and nyquist bins
    print(H.shape)


    freqs = np.fft.rfftfreq(params['block_length'], d=1 / params['fs_record'])[1:-1]

    plt.figure(figsize=(10,4))

    plt.plot(
        freqs,
        20 * np.log10(np.abs(H) + 1e-12)
    )

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Estimated Channel Frequency Response')
    plt.grid(True)

    plt.show()


     # Next line simply assumes that the ODFM begins as soon as the key finishes
    start_index = sync_index + params['fs_record'] * params['key_length']
    
    
if __name__ == "__main__":
    params = {
        # MAYBE ADD CHIRP PARAMATERS E.G CHIRP LENGTH, START AND END FREQUENCIES - SAM
        'key_type': 'chirp', #up_down_chirp
        'repeat_key_count': 1,
        'f0': 100, #Start frequency of chirp
        'f1': 22000, #End frequency of chirp
        'block_length': 1024,
        'cyclic_prefix_length': 32,
        'read_prefix_early_samples': 5, #Deliberately read some samples before the detected sync index 
        'length_of_key': 4800, # length of key 
        'fs': 44100, #Generating signal
        'fs_record': 44100, #Recording signal
        'silence_duration': 0.0,
        'record_duration': 10, #Length of recording
        'signal_name': 'test_signal_XX.wav',
        'recording_name': 'test_recording_XX.wav'


    }
    main(params)