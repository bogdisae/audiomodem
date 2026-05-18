from scipy.io import wavfile
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt


fs_rx, rxChirp = wavfile.read("SAM-NS/recieved_chirp.wav")
fs_tx, txChirp = wavfile.read("SAM-NS/chirp.wav")

rxChirp = rxChirp.astype(np.float32)
txChirp = txChirp.astype(np.float32)

rxChirp /= np.max(np.abs(rxChirp))
txChirp /= np.max(np.abs(txChirp))


corr = correlate(rxChirp, txChirp, mode='valid')
sync_index = np.argmax(np.abs(corr))
print("Chirp starts at sample:", sync_index)


corr_abs = np.abs(corr)

top_100_indices = np.argsort(corr_abs)[-100:][::-1]
top_100_values = corr_abs[top_100_indices]

for i in range(100):
    print(top_100_indices[i], top_100_values[i])


    # Sample indices for x-axis
x = np.arange(len(corr))

# Plot
plt.figure()
plt.plot(x, np.abs(corr))
plt.title("Matched Filter Output (Correlation)")
plt.xlabel("Sample index")
plt.ylabel("Correlation magnitude")
plt.show()