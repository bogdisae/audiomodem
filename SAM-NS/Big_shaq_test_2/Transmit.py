import numpy as np
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import chirp
from functions import bytes_csv_to_bits, bits_to_qpsk, frame_symbols, ofdm_modulate, generate_chirp
from functions import build_transmit_signal

with open('SAM-NS/BIGSHAQ.txt', "r", encoding="utf-8") as f:
    text = f.read()

bit_list = bytes_csv_to_bits(text)
symbols = bits_to_qpsk(bit_list)
framed_symbols = frame_symbols(symbols, 511)

print(len(framed_symbols))

# OFDM modulation
ofdm_blocks = [
    ofdm_modulate(frame, n_fft=1024)
    for frame in framed_symbols
]

# COMBINE WITH CHIRP
chirp = generate_chirp(48000, 1, 100, 8000)

# Create a double chirp preamble (repeated pattern)
silence = np.zeros(1 * 48000)
doubleChirp = np.concatenate([chirp, silence, chirp])

fullSignal = build_transmit_signal(ofdm_blocks, 128, doubleChirp)

# Normalize ENTIRE waveform
fullSignal = fullSignal / np.max(np.abs(fullSignal))

combined_int16 = np.int16(fullSignal * 32767)
write("SAM-NS/Big_shaq_test_2/test.wav", 48000, combined_int16)