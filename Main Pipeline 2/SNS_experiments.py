from equaliser import Equaliser, RepeatedChirp
from tx import Tx
from helper import pick_text_file, csv_to_data_bytes
from pathlib import Path
import numpy as np
import questionary
from scipy.io.wavfile import write

sampleRate = 48000

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


generateRepeatedChirp()