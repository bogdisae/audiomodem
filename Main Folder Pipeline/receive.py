#import relevant libraries
from pathlib import Path
import questionary
from scipy.io import wavfile
from receive_functions import normalise_signal, key_matched_filter, record_audio, generate_key
from rx_signal import RxSignal # OUR OWN CLASS

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

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

def main(params):

    # Choose how to access the audio file

    mode = questionary.select("Do you want to record audio or select an existing file?",
        choices=["Record audio", "Select file"]
    ).ask()
    if mode is None: raise SystemExit("No option selected")

    if mode == "Select file":
        selected_path = pick_wav_file("Select a WAV file:", Path("./Main Folder Pipeline/Audio Files"))
        # Load the audio file INTO OUR OWN CLASS
        fs_rx, sig = wavfile.read(selected_path)
        rxSig = RxSignal(normalise_signal(sig))

    elif mode == "Record audio":
        print("Recording mode selected")
        sig = record_audio(params['fs'])
        rxSig = RxSignal(normalise_signal(sig))
    
    # NOW WE SYNCHRONISE!

    try:
        sync_index = key_matched_filter(rxSig, params['fs_record'], params['length_of_key'] / params['fs_record'], params['f0'], params['f1'], params['key_type'])
    except ValueError as e:
        print("Error during matched filtering:", e)
        raise

    print(f"{params['key_type']} starts at sample:", sync_index)

    end_idx = sync_index + params['length_of_key'] #Length of Chirp
    isolated_key = rxSig[sync_index:end_idx]
    Y = np.fft.rfft(isolated_key, n=params['block_length']) # Why do we use the block length as the DFT length? - Sam

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
            'length_of_key': 12000, # length of key 
            'f0': 100, #Start frequency of chirp
            'f1': 4000, #End frequency of chirp
            'block_length': 1024,
            'cyclic_prefix_length': 128,
            'read_prefix_early_samples': 30, # Deliberately read some samples before the detected sync index 
            'fs': 48000, # GLOBAL sample rate
        }
    main(params)