#import relevant libraries
from scipy.io.wavfile import write
from pathlib import Path
import questionary
from transmit_functions import bytes_csv_to_bits, bits_to_qpsk, frame_symbols, ofdm_modulate, build_transmit_signal, Constellation
from receive_functions import generate_key
import numpy as np

def pick_text_file(prompt_text: str, folder: Path) -> str:
    txt_files = sorted(folder.glob('*.txt'))
    
    if not txt_files:
        raise FileNotFoundError(f'No .txt files found in {folder}')

    choice = questionary.select(
        prompt_text,
        choices=[path.name for path in txt_files],
    ).ask()

    if choice is None:
        raise SystemExit('No file selected')

    return str(folder / choice)

def main(params):

    text_file = pick_text_file("Select message file:", Path("./Main Folder Pipeline/Data Files"))
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    constellation = Constellation(2, {
        (0, 0): (1+1j),
        (0, 1): (-1+1j),
        (1, 0): (1-1j),
        (1, 1): (-1-1j)
    })

    # Convert data to blocks (frames) of bits
    bit_list = bytes_csv_to_bits(text)
    symbols = bits_to_qpsk(bit_list, constellation)            # This is assuming qpsk modulation! Maybe add a paramater for the modulation type
    num_Info_Symbols = (params['block_length'] // 2) - 1    # E.g only 511 symbols for a 1024 length block
    framed_symbols = frame_symbols(symbols, num_Info_Symbols)    

    # OFDM modulation
    ofdm_blocks = [ofdm_modulate(frame, n_fft=params['block_length']) for frame in framed_symbols]
    print("Length of OFDM blocks:", len(ofdm_blocks[0]))
    assert all(np.max(np.abs(np.imag(block))) == 0 for block in ofdm_blocks), \
    "Complex values detected in OFDM blocks"
    
    key = generate_key(params['fs'], params['length_of_key']/params['fs'], params['f0'], params['f1'], params['key_type'])

    fullSignal = build_transmit_signal(ofdm_blocks, params['cyclic_prefix_length'], key)
    combined_int16 = np.int16(fullSignal * 32767) # Convert to wav amplitudes

    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Folder Pipeline/Audio Files/{filename}.wav", params['fs'], combined_int16)


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