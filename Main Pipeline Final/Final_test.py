print("Importing modules...")

import tempfile

from equaliser import Equaliser, RepeatedChirp, GolayPairs, WhiteNoise
from tx import Tx
from rx import Rx
from helper import *
from file_decode_func import *
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

def generate_final_sig(standard = True):
    #Pick file path to desired file
    transmitted_file = pick_file("Select file:", Path("./Main Pipeline Final/Data Files"))
    file_name_tx = Path(transmitted_file).name  
    print(f"Selected file: {transmitted_file}")
    data_bytes = file_to_numpy(transmitted_file)

    #Pass data_bytes to Tx, get transmitted signal
    repeatedChirp = RepeatedChirp(10, 4096, 0, 750, 18000, sync = True, est = False, fs = sampleRate)
    golayPairs = GolayPairs(12, silence = 2048, numPairs=4, seed = (1,1), est = True, fs = 48000) #2**12 = 4096
    whiteNoise = WhiteNoise(4096, 2048,  constellation, sync = False, est = True, fs = sampleRate)
    transmitter = Tx(
        constellation=constellation,
        header_filename = file_name_tx,
        data_bytes = data_bytes,
        equaliser1 = repeatedChirp,
        equaliser2 = golayPairs,
        equaliser3 = whiteNoise,
        cp_length = 2048,
        block_length = 4096,
        pilot_spacing = 20,
        f_low = 2000,
        f_high = 12000,
        use_ldpc = True
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
    golayPairs = GolayPairs(12, silence = 2048, numPairs=4, seed = (1,1), est = True, fs = sampleRate) #2**12 = 4096
    whiteNoise = WhiteNoise(4096, 2048,constellation, sync = False, est = False, fs = sampleRate)
    
    # LIST SHOULD ONLY CONTAIN THE INITIAL REPEATED CHIRP AND GOLAY SEQUENCE!!
    equaliserList = [repeatedChirp, golayPairs]

    # DON'T KNOW WHAT SFO EQUALISER ACTUALLY IS YET
    receiver = Rx(
        constellation = constellation,
        signal = sig,
        cp_length = 2048,
        block_length = 4096,
        equalisers = equaliserList,
        sfoEqualiser = None,
        use_ldpc = True
    )
    receiver.decode()

    #SAVE DECODED BYTES TO FILE
    data_bytes = receiver.payload
    try:
        temp_filename = receiver.filename

    except:
        temp_filename = "Output_Name_Unknown"
    print(f"Decoded filename: {temp_filename}")

    file_path = Path(f"Main Pipeline Final/Decoded Files/{temp_filename}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(data_bytes.tobytes())

    #Post receive stats
    print("Number of symbols", len(receiver.data_symbols))


        
#generate_final_sig()

receive_standard_sig()


