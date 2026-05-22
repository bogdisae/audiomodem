#import relevant libraries
from pathlib import Path
import questionary
from scipy.io import wavfile
from receive_functions import normalise_signal, key_synchronise, record_audio, generate_key, wiener_filter_coeffs, demodulate_ofdm_signal
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
        rxSig.keyIdxStart = key_synchronise(rxSig, params['fs'], params['length_of_key'] / params['fs'], params['f0'], params['f1'], params['key_type'])
    except ValueError as e:
        print("Error during matched filtering:", e)
        raise

    print(f"{params['key_type']} starts at sample:", rxSig.keyIdxStart)

    rxSig.keyIdxEnd = rxSig.keyIdxStart + params['length_of_key'] 
    isolated_key = rxSig.sigArray[rxSig.keyIdxStart : rxSig.keyIdxEnd]

    DFT_LENGTH = 12000  # THIS IS IMPORTANT! THINK ABOUT HOW MANY DFT POINTS YOU ACTUALLY NEED (E.G NUMBER OF SAMPLES IN CHIRP)
    # THIS WILL NOT WORK IN THE DFT PIPELINE UNTIL WE HAVE 1024 COEFFICIENTS

    Y = np.fft.rfft(isolated_key, n = DFT_LENGTH)
    key = generate_key(params['fs'], params['length_of_key'] / params['fs'], params['f0'], params['f1'], params['key_type'])
    S = np.fft.rfft(key, n = DFT_LENGTH)

    eps = 1e-12  # Prevent divide-by-zero instability
    H = Y[1:-1] / (S[1:-1] + eps) # Remove DC and nyquist bins

    # TEMPORARY TEST CODE
    H_1024 = np.interp(np.linspace(0, len(H)-1, 1024), np.arange(len(H)), H)

    wiener_filter_coeffs(isolated_key, key, filter_N=1024, fs = params['fs'], plotting=True)

    freqs = np.fft.rfftfreq(DFT_LENGTH, d=1 / params['fs'])[1:-1]
    plt.figure(figsize=(10,4))
    plt.plot(freqs, 20 * np.log10(np.abs(H) + 1e-12))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Estimated Channel Frequency Response')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(np.arange(1024), 20*np.log10(np.abs(H_1024) + 1e-12))
    plt.xlabel("Subcarrier index")
    plt.ylabel("Magnitude (dB)")
    plt.title("OFDM Channel Response (1024 subcarriers)")
    plt.grid(True)
    plt.show()


     # Next line simply assumes that the ODFM begins as soon as the key finishes
    rxSig.dataIdx = rxSig.keyIdxStart + params['length_of_key']



    #demodulate_ofdm_signal(equalised_signal)
    
    
if __name__ == "__main__":

    params = {
            # MAYBE ADD CHIRP PARAMATERS E.G CHIRP LENGTH, START AND END FREQUENCIES - SAM
            'key_type': 'chirp', #up_down_chirp
            'length_of_key': 48000, # length of key 
            'f0': 0, #Start frequency of chirp
            'f1': 20000, #End frequency of chirp
            'block_length': 1024,
            'cyclic_prefix_length': 128,
            'read_prefix_early_samples': 30, # Deliberately read some samples before the detected sync index 
            'fs': 48000, # GLOBAL sample rate
            'modulation_scheme': 'QPSK'
        }
    main(params)