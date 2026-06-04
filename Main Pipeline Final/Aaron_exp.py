print("Importing modules...")
import os

from constellation import *
from equaliser import *
from helper import *
from tx import *
from rx import *

from pathlib import Path
import numpy as np
import questionary
from scipy.io import wavfile

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
    ('0',): 1,
    ('1',): -1
}, {
    ('0',): lambda s: s.real >= 0,
    ('1',): lambda s: s.real < 0
})

def generate_sig(standard = True):
    text_file = pick_csv_file("Select message file:", Path("./Main Pipeline 2/Data Files"))
    data_bytes = csv_to_data_bytes(text_file)


    #STANDARD CHIRP PARAMETERS
    standard == True
    if standard == True:
        print("Using standard chirp parameters for testing")
        repeatedChirp = RepeatedChirp(10, 4096, 0, 750, 18000, sync = True, fs = sampleRate)
        golayPairs = GolayPairs(12, silence = 2048, numPairs=4, seed = (1,1), est = True, fs = 48000) #2**12 = 4096
        WN = np.zeros(4096)

        transmitter = Tx(
            constellation=constellation,
            data_bytes=data_bytes,
            equaliser1 = repeatedChirp,
            equaliser2 = golayPairs,
            equaliser3 = WN,
            cp_length = 2048,
            block_length = 4096,
            pilot_spacing = 20,
            f_low = 2000,
            f_high = 12000
        )

    transmitter.encode()

    plot_constellation(transmitter.data_symbols[-2000:], "Transmitted Constellation")

    sig = transmitter.transmitted_signal

    combined_int16 = np.int16(sig * 32767) # Convert to wav amplitudes
    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Pipeline Final/Audio Files/{filename}.wav", sampleRate, combined_int16)
    print(f'Saved in dir: Main Pipeline Final/Audio Files/{filename}.wav')

generate_sig()