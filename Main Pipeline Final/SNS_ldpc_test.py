from ldpc import ldpc
import numpy as np
from pathlib import Path

from helper import pick_text_file, csv_to_data_bytes

c = ldpc.code('802.16', z=61)
text_file = pick_text_file("Select message file:", Path("./Main Pipeline Final/Data Files"))
data_bytes = csv_to_data_bytes(text_file)
data_bits = np.unpackbits(data_bytes)
pad = (-len(data_bits)) % c.K
bits_padded = np.pad(data_bits, (0, pad))

print("bits_padded % c.K should = 0:", len(bits_padded)%c.K)

blocks = bits_padded.reshape(-1, c.K)
print("Shape of blocks array", blocks.shape)
encoded = np.array([c.encode(block) for block in blocks])
print("Shape of encoded ldpc:", encoded.shape)

llr_input = np.array([1000, -1000])[encoded]
decoded = np.array([c.decode(block)[0] for block in llr_input])
data_bits_received = (decoded < 0).astype(np.uint8)
data_bits_received = data_bits_received[:, :c.K].flatten()
data_bits_received = data_bits_received[:len(data_bits)]

print(np.array_equal(data_bits, data_bits_received))
print(data_bits[:100])
print(data_bits_received[:100])


hard = (decoded < 0).astype(np.uint8)

print("Original first block:")
print(blocks[0][:50])

print("Decoded first block:")
print(hard[0][:50])

print("Bit errors:",
      np.sum(blocks != hard[:, :c.K]))