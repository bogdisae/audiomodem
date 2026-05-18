from scipy.io import wavfile
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt


def main(wav_file_1, wav_file_2):
    fs_rx, rxChirp = wavfile.read(wav_file_1)
    fs_tx, txChirp = wavfile.read(wav_file_2)

    rxChirp = rxChirp.astype(np.float32)
    txChirp = txChirp.astype(np.float32)

    rxChirp /= np.max(np.abs(rxChirp))
    txChirp /= np.max(np.abs(txChirp))


    corr = correlate(rxChirp, txChirp, mode='valid')
    sync_index = np.argmax(np.abs(corr))
    print("Chirp starts at sample:", sync_index)

    # Sample indices for x-axis
    x = np.arange(len(corr))

    # Plot
    plt.figure()
    plt.plot(x, np.abs(corr))
    plt.title("Matched Filter Output (Correlation)")
    plt.xlabel("Sample index")
    plt.ylabel("Correlation magnitude")
    plt.show()


    # Sample indices for x-axis
    x = np.arange(len(corr))

    # Plot
    plt.figure()
    plt.plot(x, corr)
    plt.title("Matched Filter Output (Correlation)")
    plt.xlabel("Sample index")
    plt.ylabel("Correlation magnitude")
    plt.show()

if __name__ == "__main__":
    main()