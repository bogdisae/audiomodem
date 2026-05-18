import numpy as np
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import chirp
from functions import bytes_csv_to_bits, bits_to_qpsk, frame_symbols, ofdm_modulate, generate_chirp


with open('SAM-NS/BIGSHAQ.txt', "r", encoding="utf-8") as f:
    text = f.read()

bit_list = bytes_csv_to_bits(text)
symbols = bits_to_qpsk(bit_list)
framed_symbols = frame_symbols(symbols, 511)

# Now lets take the first block of symbols and perform the IDFT
first_block_signal = ofdm_modulate(framed_symbols[0], 1024)

# # COMBINE WITH CHIRP
chirp = generate_chirp(48000, 1, 100, 8000)

first_signal_repeated = np.tile(first_block_signal.real, 500)
first_signal_repeated = first_signal_repeated / np.max(np.abs(first_signal_repeated))

silence = np.zeros(2 * 48000)
combined = np.concatenate([chirp, silence, first_signal_repeated])

combined_int16 = np.int16(combined * 32767)
write("SAM-NS/test.wav", 48000, combined_int16)