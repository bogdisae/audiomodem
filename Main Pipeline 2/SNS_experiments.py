from equaliser import Equaliser, RepeatedChirp, Chirp
from tx import Tx
from rx import Rx
from helper import pick_text_file, csv_to_data_bytes, pick_wav_file, normalise_signal, record_audio, plot_constellation
from helper import csv_bytes_to_binary_sequence, calculate_ber
from pathlib import Path
import numpy as np
import questionary
from scipy.io import wavfile
from scipy.io.wavfile import write
from constellation import Constellation

sampleRate = 48000

# constellation = Constellation(2, {
#     ('0', '0'): (1+1j)/np.sqrt(2),
#     ('0', '1'): (-1+1j)/np.sqrt(2),
#     ('1', '0'): (1-1j)/np.sqrt(2),
#     ('1', '1'): (-1-1j)/np.sqrt(2)
# }, {
#     ('0', '0'): lambda s: (s.real >= 0) & (s.imag >= 0),
#     ('0', '1'): lambda s: (s.real < 0) & (s.imag >=  0),
#     ('1', '0'): lambda s: (s.real >=  0) & (s.imag < 0),
#     ('1', '1'): lambda s: (s.real <  0) & (s.imag <  0),
# })

constellation = Constellation(1, {
    ('0',): 1+0j,
    ('1',): -1+0j,
}, {
    ('0',): lambda s: (s.real >= 0),
    ('1',): lambda s: (s.real < 0),
})


def generateRepeatedChirp_plus_data():
    text_file = pick_text_file("Select message file:", Path("./Main Pipeline 2/Data Files"))
    data_bytes = csv_to_data_bytes(text_file)

    repeatedChirp = RepeatedChirp(10, 1024, 0, 20, 20000, sampleRate)
    key = repeatedChirp.generate()

    transmitter = Tx(constellation, data_bytes, repeatedChirp, 1024, 1024)
    transmitter.encode()

    sig = transmitter.transmitted_signal

    combined_int16 = np.int16(sig * 32767) # Convert to wav amplitudes
    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Pipeline 2/Audio Files/{filename}.wav", sampleRate, combined_int16)
    
def receiveRepeated_chirp_plus_data():

    mode = questionary.select("Do you want to record audio or select an existing file?",
        choices=["Record audio", "Select file"]
    ).ask()

    if mode is None: raise SystemExit("No option selected")

    if mode == "Select file":
        selected_path = pick_wav_file("Select a WAV file:", Path("./Main Pipeline 2/Audio Files"))
        fs_rx, sig = wavfile.read(selected_path)
        sig = normalise_signal(sig)

    elif mode == "Record audio":
        print("Recording mode selected")
        sig = record_audio(sampleRate)
        sig = normalise_signal(sig)
    

    repeatedChirp = RepeatedChirp(10, 1024, 0, 20, 20000, sampleRate)
    receiver = Rx(constellation, sig, 1024, 1024, repeatedChirp)
    receiver.decode()
    print ("Number of coefficients:", len(receiver.H))
    print("First 10 estimated coefficients:\n", receiver.H[:10])

    print(receiver.data_bits[:200])
    plot_constellation(receiver.data_symbols[:200])
    plot_constellation(receiver.data_symbols[0:2000])
    # plot_constellation(receiver.data_symbols[2000:4000])
    # plot_constellation(receiver.data_symbols[4000:6000])
    # plot_constellation(receiver.data_symbols[6000:8000])
    # plot_constellation(receiver.data_symbols[8000:10000])
    # plot_constellation(receiver.data_symbols[10000:12000])
    # plot_constellation(receiver.data_symbols[12000:14000])
    # plot_constellation(receiver.data_symbols[14000:16000])
    print("Number of data symbols:", len(receiver.data_symbols))



    shaqbits = csv_bytes_to_binary_sequence("Main Pipeline 2/Data Files/BIGSHAQ.txt")
    ber, errors, min_len = calculate_ber(shaqbits, receiver.data_bits[:2000])

    print("BER:", ber)
    print("Errors:", errors)
    print("Min Len", min_len)

    print(receiver.data_bytes)

def generateSingleChirp_plus_data():
    
    text_file = pick_text_file("Select message file:", Path("./Main Pipeline 2/Data Files"))
    data_bytes = csv_to_data_bytes(text_file)

    chirp = Chirp(0, 20000, 24000) # CHIRP LENGTH IN SAMPLES?? 
    key = chirp.generate()

    transmitter = Tx(constellation, data_bytes, chirp, 1024, 1024)
    transmitter.encode()

    sig = transmitter.transmitted_signal

    combined_int16 = np.int16(sig * 32767) # Convert to wav amplitudes
    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Pipeline 2/Audio Files/{filename}.wav", sampleRate, combined_int16)

def receive_SingleChirp_plus_data(): # DOESNT WORK BECAUSE THE CHANNEL ESTIMATION IS FUCKED!

    mode = questionary.select("Do you want to record audio or select an existing file?",
        choices=["Record audio", "Select file"]
    ).ask()

    if mode is None: raise SystemExit("No option selected")

    if mode == "Select file":
        selected_path = pick_wav_file("Select a WAV file:", Path("./Main Pipeline 2/Audio Files"))
        fs_rx, sig = wavfile.read(selected_path)
        sig = normalise_signal(sig)

    elif mode == "Record audio":
        print("Recording mode selected")
        sig = record_audio(sampleRate)
        sig = normalise_signal(sig)
    

    chirp = Chirp(0, 20000, 24000)
    receiver = Rx(constellation, sig, 1024, 1024, chirp)
    receiver.decode()


    print(receiver.data_bits[:200])
    plot_constellation(receiver.data_symbols[0:2000])

    shaqbits = csv_bytes_to_binary_sequence("Main Pipeline 2/Data Files/BIGSHAQ.txt")
    ber, errors, min_len = calculate_ber(shaqbits, receiver.data_bits[:5000])

    print("BER:", ber)
    print("Errors:", errors)
    print("Min Len", min_len)

    print(receiver.data_bytes)


receiveRepeated_chirp_plus_data()
#generateRepeatedChirp_plus_data()

#generateSingleChirp_plus_data()
#receive_SingleChirp_plus_data()


