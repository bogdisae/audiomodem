import numpy as np
from scipy.signal import correlate, chirp
from scipy.io import wavfile
import matplotlib.pyplot as plt


def normalise_signal(signal):
    signal = signal.astype(np.float32)

    max_val = np.max(np.abs(signal))

    if max_val == 0:
        return signal

    return signal / max_val

def generate_chirp(fs, T, f0, f1):
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    chirp_signal = chirp(t, f0, f1, T, method='linear')
    chirp_signal = chirp_signal / np.max(np.abs(chirp_signal))
    return chirp_signal

def chirp_matched_filter(signal, fs, T, f0, f1):
    signal = np.asarray(signal).squeeze()
    chirp_sig = generate_chirp(fs, T, f0, f1).squeeze()

    corr = correlate(signal, chirp_sig, mode='valid')
    sync_index = np.argmax(np.abs(corr))

    # Sample indices for x-axis
    x = np.arange(len(corr))
    # Plot
    plt.figure()
    plt.plot(x, np.abs(corr))
    plt.title("Matched Filter Output - Absolute value")
    plt.xlabel("Sample index")
    plt.ylabel("Correlation magnitude")
    plt.show()


    return sync_index