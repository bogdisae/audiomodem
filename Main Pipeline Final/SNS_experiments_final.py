from equaliser import Equaliser, RepeatedChirp, GolayPairs, WhiteNoise
from tx import Tx
from rx import Rx
from helper import pick_text_file, csv_to_data_bytes, pick_wav_file, normalise_signal, record_audio, plot_constellation
from helper import csv_bytes_to_binary_sequence, calculate_ber
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import questionary
from scipy.io import wavfile
from scipy.io.wavfile import write
from constellation import Constellation

sampleRate = 48000

constellation = Constellation(2, {
    ('0', '0'): (1+1j)/np.sqrt(2),
    ('0', '1'): (-1+1j)/np.sqrt(2),
    ('1', '0'): (1-1j)/np.sqrt(2),
    ('1', '1'): (-1-1j)/np.sqrt(2)
}, {
    ('0', '0'): lambda s: (s.real >= 0) & (s.imag >= 0),
    ('0', '1'): lambda s: (s.real < 0) & (s.imag >=  0),
    ('1', '0'): lambda s: (s.real >=  0) & (s.imag < 0),
    ('1', '1'): lambda s: (s.real <  0) & (s.imag <  0),
})

# constellation = Constellation(1, {
#     ('0',): 1+0j,
#     ('1',): -1+0j,
# }, {
#     ('0',): lambda s: (s.real >= 0),
#     ('1',): lambda s: (s.real < 0),
# })



def generate_standard_sig(standard = True):
    text_file = pick_text_file("Select message file:", Path("./Main Pipeline Final/Data Files"))
    data_bytes = csv_to_data_bytes(text_file)

    repeatedChirp = RepeatedChirp(10, 4096, 0, 750, 18000, sync = True, est = True, fs = sampleRate)
    golayPairs = GolayPairs(12, silence = 2048, numPairs=4, seed = (1,1), est = True, fs = 48000) #2**12 = 4096
    whiteNoise = WhiteNoise(4096, constellation, sync = False, est = True, fs = sampleRate)
    transmitter = Tx(
        constellation=constellation,
        data_bytes = data_bytes,
        equaliser1 = repeatedChirp,
        equaliser2 = golayPairs,
        equaliser3 = whiteNoise,
        cp_length = 2048,
        block_length = 4096,
        pilot_spacing = 20,
        f_low = 2000,
        f_high = 12000
        )

    transmitter.encode()

    sig = transmitter.transmitted_signal

    combined_int16 = np.int16(sig * 32767) # Convert to wav amplitudes
    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Pipeline Final/Audio Files/{filename}.wav", sampleRate, combined_int16)

        
def receive_standard_sig():

    mode = questionary.select("Do you want to record audio or select an existing file?",
        choices=["Record audio", "Select file"]
    ).ask()

    if mode is None: raise SystemExit("No option selected")

    if mode == "Select file":
        selected_path = pick_wav_file("Select a WAV file:", Path("./Main Pipeline Final/Audio Files"))
        fs_rx, sig = wavfile.read(selected_path)
        sig = normalise_signal(sig)

    elif mode == "Record audio":
        print("Recording mode selected")
        sig = record_audio(sampleRate)
        sig = normalise_signal(sig)
        
    
    repeatedChirp = RepeatedChirp(10, 4096, 0, 750, 18000, sync = True, est = True, fs = sampleRate)
    golayPairs = GolayPairs(12, silence = 2048, numPairs=4, seed = (1,1), est = False, fs = sampleRate) #2**12 = 4096
    whiteNoise = WhiteNoise(4096, constellation, sync = False, est = False, fs = sampleRate)

    # LIST SHOULD ONLY CONTAIN THE INITIAL REPEATED CHIRP AND GOLAY SEQUENCE!!
    equaliserList = [repeatedChirp, golayPairs]

    # DON'T KNOW WHAT SFO EQUALISER ACTUALLY IS YET
    receiver = Rx(constellation, sig, 2048, 4096, equaliserList, None)
    receiver.decode()

    print("Receiver preamble start estimates:", receiver.preamble_start_estimates)
    print("Receiver key start estimates:", receiver.key_start_estimates)
    print("Data start estimate:" , receiver.data_start_estimate)
    print("Index where decoding starts:", receiver.decode_start)

    


    # print ("Number of coefficients:", len(receiver.H))
    # print("First 10 estimated coefficients:\n", receiver.H[:10])

    # print(receiver.data_bits[:200])
    plot_constellation(receiver.data_symbols[:200])
    plot_constellation(receiver.data_symbols[0:4000])
    # plot_constellation(receiver.data_symbols[2000:4000])
    # plot_constellation(receiver.data_symbols[4000:6000])
    # plot_constellation(receiver.data_symbols[6000:8000])
    # plot_constellation(receiver.data_symbols[8000:10000])
    # plot_constellation(receiver.data_symbols[10000:12000])
    # plot_constellation(receiver.data_symbols[12000:14000])
    # plot_constellation(receiver.data_symbols[14000:16000])
    # print("Number of data symbols:", len(receiver.data_symbols))



    shaqbits = csv_bytes_to_binary_sequence("Main Pipeline Final/Data Files/BIGSHAQ.txt")
    ber, errors, min_len = calculate_ber(shaqbits, receiver.data_bits[:200])

    print("BER:", ber)
    print("Errors:", errors)
    print("Min Len", min_len)

    # print(receiver.data_bytes)


    # print ("Number of coefficients:", len(receiver.H))
    # print("First 10 estimated coefficients:\n", receiver.H[:10])

    # print(receiver.data_bits[:200])
    # plot_constellation(receiver.data_symbols[:200])
    # plot_constellation(receiver.data_symbols[0:2000])
    # plot_constellation(receiver.data_symbols[2000:4000])
    # plot_constellation(receiver.data_symbols[4000:6000])
    # plot_constellation(receiver.data_symbols[6000:8000])
    # plot_constellation(receiver.data_symbols[8000:10000])
    # plot_constellation(receiver.data_symbols[10000:12000])
    # plot_constellation(receiver.data_symbols[12000:14000])
    # plot_constellation(receiver.data_symbols[14000:16000])
    # print("Number of data symbols:", len(receiver.data_symbols))



    # shaqbits = csv_bytes_to_binary_sequence("Main Pipeline 2/Data Files/BIGSHAQ.txt")
    # ber, errors, min_len = calculate_ber(shaqbits, receiver.data_bits[:2000])

    # print("BER:", ber)
    # print("Errors:", errors)
    # print("Min Len", min_len)

    # print(receiver.data_bytes)


#generate_standard_sig()

receive_standard_sig()

