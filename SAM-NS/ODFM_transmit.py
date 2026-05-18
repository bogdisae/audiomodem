import numpy as np
import numpy as np
from scipy.io.wavfile import write

with open('SAM-NS/BIGSHAQ.txt', "r", encoding="utf-8") as f:
    text = f.read()

byte_list = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
bit_list = []
for byte in byte_list:
    bits = format(byte, "08b")
    bit_list.extend(int(b) for b in bits)

# Make sure it is divisible by 2
bit_list = bit_list[:len(bit_list) - (len(bit_list) % 2)]

symbols = np.array(
    [
        (1 + 1j) if (bit_list[i], bit_list[i+1]) == (0, 0) else
        (-1 + 1j) if (bit_list[i], bit_list[i+1]) == (0, 1) else
        (-1 - 1j) if (bit_list[i], bit_list[i+1]) == (1, 1) else
        (1 - 1j)
        for i in range(0, len(bit_list), 2)
    ],
    dtype=complex
) / np.sqrt(2)

blocks = [symbols[i:i+511] for i in range(0, len(symbols), 511)]

block = blocks[3]

X = np.zeros(1024, dtype=complex)

X[1:512] = block[:511]

X[513:] = np.conj(X[1:512][::-1])

x = np.fft.ifft(X)

# Repeat to form the real time domain signal
x_repeated = np.tile(x.real, 500)
x_repeated = x_repeated / np.max(np.abs(x_repeated))
fs = 48000
write("SAM-NS/test.wav", fs, x_repeated.astype(np.float32))

