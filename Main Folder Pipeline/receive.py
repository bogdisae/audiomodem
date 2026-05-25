#import relevant libraries
from pathlib import Path
import questionary
from scipy.io import wavfile
from receive_functions import compare_wiener_length, normalise_signal, key_synchronise, record_audio, generate_key, wiener_filter_coeffs, demodulate_ofdm_signal
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

# TEMPORARY FUNCTIONS
def estimate_channel_per_chirp(rx_signal, tx_chirp, start_idx, chirp_len, stride, num_chirps):

    # FFT of known transmitted chirp
    X = np.fft.fft(tx_chirp, n=chirp_len)

    H_list = []

    for i in range(num_chirps):

        start = start_idx + i * stride
        print(i, start)
        segment = rx_signal[start:start + chirp_len]
        
        # FFT of received chirp
        Y = np.fft.fft(segment, n=chirp_len)

        eps = 1e-12

        H = Y / (X + eps)

        H_list.append(H)

    H_list = np.array(H_list)

    H_avg = np.mean(H_list, axis=0)

    return H_list, H_avg

def plot_channel_estimates(H_list, H_avg):

    num = len(H_list)

    plt.figure(figsize=(12, 6))

    for i in range(num):
        plt.plot(20*np.log10(np.abs(H_list[i]) + 1e-12),
                 alpha=0.5,
                 label=f"chirp {i}" if i < 5 else None)

    plt.plot(20*np.log10(np.abs(H_avg) + 1e-12),
             color='black',
             linewidth=2,
             label="AVERAGE")

    plt.title("Channel Estimates per Chirp + Average")
    plt.xlabel("Subcarrier index")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    plt.legend()
    plt.show()

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

    # DFT_LENGTH = 12000  # THIS IS IMPORTANT! THINK ABOUT HOW MANY DFT POINTS YOU ACTUALLY NEED (E.G NUMBER OF SAMPLES IN CHIRP)
    # # THIS WILL NOT WORK IN THE DFT PIPELINE UNTIL WE HAVE 1024 COEFFICIENTS
    
    # Y = np.fft.rfft(isolated_key, n = DFT_LENGTH)
    # key = generate_key(params['fs'], params['length_of_key'] / params['fs'], params['f0'], params['f1'], params['key_type'])

    # S = np.fft.rfft(key, n = DFT_LENGTH)

    # eps = 1e-12  # Prevent divide-by-zero instability
    # H = Y[1:-1] / (S[1:-1] + eps) # Remove DC and nyquist bins

    # # TEMPORARY TEST CODE
    # H_1024 = np.interp(np.linspace(0, len(H)-1, 1024), np.arange(len(H)), H)


    # freqs = np.fft.rfftfreq(DFT_LENGTH, d=1 / params['fs'])[1:-1]
    # plt.figure(figsize=(10,4))
    # plt.plot(freqs, 20 * np.log10(np.abs(H) + 1e-12))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude (dB)')
    # plt.title('Estimated Channel Frequency Response')
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(10,4))
    # plt.plot(np.arange(1024), 20*np.log10(np.abs(H_1024) + 1e-12))
    # plt.xlabel("Subcarrier index")
    # plt.ylabel("Magnitude (dB)")
    # plt.title("OFDM Channel Response (1024 subcarriers)")
    # plt.grid(True)
    # plt.show()


    # #h_coeffs = wiener_filter_coeffs(isolated_key, key, filter_N=500, fs = params['fs'], plotting=True)

    
    # '''#AARON WILL FIX SATURDAY
    # run_comparison = True
    # if run_comparison:
    #     #compare_wiener_length(isolated_key, key, params['fs'])
    #     compare_wiener_length(
    #         isolated_key,
    #         key,
    #         params['fs'],
    #         plot_options={
    #         "compare_all_filters": True,
    #         "show_db": True
    #         }
    #     )'''
    


    #  # Next line simply assumes that the ODFM begins as soon as the key finishes
    rxSig.dataIdx = rxSig.keyIdxStart + params['length_of_key']

    # demodulate_ofdm_signal(params, rxSig.sigArray, H_1024, rxSig.dataIdx)

    chirp_len = 1024
    silence_len = 1376
    stride = chirp_len + silence_len
    num_chirps = 10  # or infer from params
    chirps = []

    for i in range(num_chirps):

        start = rxSig.dataIdx + i * stride
        end = start + chirp_len

        segment = rxSig.sigArray[start:end]

        chirps.append(segment)

    tx_chirp = generate_key(48000, 1024 / 48000, 0, 20000, 'chirp')

    H_list, H_avg = estimate_channel_per_chirp(
        rxSig.sigArray,
        tx_chirp,
        rxSig.keyIdxStart,
        chirp_len,
        stride,
        num_chirps
    )

    plot_channel_estimates(H_list, H_avg)

    demodulate_ofdm_signal(params, rxSig.sigArray, H_avg, rxSig.dataIdx)

    
if __name__ == "__main__":

    params = {
            # MAYBE ADD CHIRP PARAMATERS E.G CHIRP LENGTH, START AND END FREQUENCIES - SAM
            'key_type': 'repeat_chirp_0.5s', #up_down_chirp
            'length_of_key': 24000, # length of key 
            'f0': 0, #Start frequency of chirp
            'f1': 20000, #End frequency of chirp
            'block_length': 1024,
            'cyclic_prefix_length': 128,
            'read_prefix_early_samples': 10, # Deliberately read some samples before the detected sync index 
            'fs': 48000, # GLOBAL sample rate
            'modulation_scheme': 'QPSK'
        }
    main(params)