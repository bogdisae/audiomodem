from equaliser import Equaliser, RepeatedChirp
from tx import Tx
from rx import Rx
from helper import pick_text_file, csv_to_data_bytes, pick_wav_file, normalise_signal, record_audio
from pathlib import Path
import numpy as np
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

def generateRepeatedChirp():
    text_file = pick_text_file("Select message file:", Path("./Main Pipeline 2/Data Files"))
    data_bytes = csv_to_data_bytes(text_file)

    repeatedChirp = RepeatedChirp(10, 1024, 1376, 0, 20000, sampleRate)
    sig = repeatedChirp.generate()

    #transmitter = Tx(None, data_bytes, repeatedChirp, 128, 1024)

    combined_int16 = np.int16(sig * 32767) # Convert to wav amplitudes

    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Pipeline 2/Audio Files/{filename}.wav", sampleRate, combined_int16)


def generateRepeatedChirp_plus_data():
    text_file = pick_text_file("Select message file:", Path("./Main Pipeline 2/Data Files"))
    data_bytes = csv_to_data_bytes(text_file)

    repeatedChirp = RepeatedChirp(10, 1024, 1376, 0, 20000, sampleRate)
    key = repeatedChirp.generate()

    transmitter = Tx(constellation, data_bytes, repeatedChirp, 128, 1024)
    transmitter.encode()

    # concatenate sync key + OFDM payload
    sig = np.concatenate([key, transmitter.transmitted_signal])

    combined_int16 = np.int16(sig * 32767) # Convert to wav amplitudes
    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Pipeline 2/Audio Files/{filename}.wav", sampleRate, combined_int16)

# generateRepeatedChirp_plus_data()

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
    

    repeatedChirp = RepeatedChirp(10, 1024, 1376, 0, 20000, sampleRate)
    receiver = Rx(constellation, sig, 128, 1024, repeatedChirp)
    receiver.decode()

    print(receiver.data_bytes)

receiveRepeated_chirp_plus_data()