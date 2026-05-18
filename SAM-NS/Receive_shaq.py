from scipy.io import wavfile
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


fs_rx, rxSig = wavfile.read("SAM-NS/big_shaq_received_3.wav")
fs_tx, txChirp = wavfile.read("SAM-NS/chirp.wav")

rxSig = rxSig.astype(np.float32)
txChirp = txChirp.astype(np.float32)

rxSig /= np.max(np.abs(rxSig))
txChirp /= np.max(np.abs(txChirp))

corr = correlate(rxSig, txChirp, mode='valid')
sync_index = np.argmax(np.abs(corr))
print("Chirp starts at sample:", sync_index)

# Sample indices for x-axis
x = np.arange(len(corr))

# Plot
plt.figure()
plt.plot(x, np.abs(corr))
plt.title("Matched Filter Output - Absolute value")
plt.xlabel("Sample index")
plt.ylabel("Correlation magnitude")
plt.show()

ofdm_start = sync_index + 2 * 96000 - 30 # Subtract 30 for safety

received_first_block = rxSig[ofdm_start: ofdm_start + 1024]

dft = np.fft.fft(received_first_block, 1024)
dft = dft[1:512]

x = np.asarray(dft.real)
y = np.asarray(dft.imag)

quadrants = np.where(
    (x >= 0) & (y >= 0),
    0,
    np.where(
        (x < 0) & (y >= 0),
        1,
        np.where((x < 0) & (y < 0), 2, 3)
    )
)

quadrant_colors = np.array(["tab:blue", "tab:orange", "tab:green", "tab:red"])
point_colors = quadrant_colors[quadrants]

plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=point_colors, s=18, alpha=0.85, edgecolors="none")

plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Argand Diagram of Equalised Data")
plt.grid()
plt.axis('equal')
plt.show()

