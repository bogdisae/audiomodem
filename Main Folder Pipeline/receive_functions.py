import numpy as np
from scipy.signal import correlate, chirp
import matplotlib.pyplot as plt
from transmit_functions import save_wav_file
import sounddevice as sd

def normalise_signal(signal):
    signal = signal.astype(np.float32)

    max_val = np.max(np.abs(signal))

    if max_val == 0:
        return signal

    return signal / max_val

## THIS FUNCTION IS IDENTICAL TO THE ONE IN TRANSMIT_FUNCTIONS. BUT I AM LAZY AND IS NICER TO SPLIT THE FUNCTIONS
def generate_key(fs, T, f0, f1, type_key='chirp'):
    if type_key == 'chirp' or type_key == 'up_down_chirp':
        t = np.linspace(0, T, int(fs * T), endpoint=False)

        signal = chirp(
            t,
            f0=f0,
            t1=T,
            f1=f1,
            method='linear'
        )

        if type_key == 'up_down_chirp':
            signal += chirp(
                t,
                f0=f1,
                t1=T,
                f1=f0,
                method='linear'
            )

    return signal / np.max(np.abs(signal))

def key_matched_filter(signal, fs, T, f0, f1, key_type='chirp', plot=True):
    #Signal input in form of numpy array

    signal = np.asarray(signal).squeeze()
    key_sig = generate_key(fs, T, f0, f1, type_key=key_type).squeeze()

    corr = correlate(signal, key_sig, mode='valid')
    sync_index = np.argmax(np.abs(corr))

    # Sample indices for x-axis
    x = np.arange(len(corr))
    # Plot
    if plot == True:
        plt.figure()
        plt.plot(x, np.abs(corr))
        plt.title("Matched Filter Output - Absolute value")
        plt.xlabel("Sample index")
        plt.ylabel("Correlation magnitude")
        plt.show()


    return sync_index


def record_audio(record_duration, fs, filename="recording.wav", channels=1):
    """
    Record audio from the default input device and save it using save_wav_file.

    Parameters
    ----------
    record_duration : float
        Recording length in seconds.
    fs : int
        Sampling frequency in Hz.
    filename : str
        Output filename passed to save_wav_file.
    channels : int
        Number of input channels to record.

    Returns
    -------
    np.ndarray
        Recorded audio as a float32 NumPy array.
    """
    print(f"Recording {record_duration} seconds at {fs} Hz...")
    recording = sd.rec(
        int(record_duration * fs),
        samplerate=fs,
        channels=channels,
        dtype="float32",
    )
    sd.wait()

    if channels == 1:
        recording = recording.reshape(-1)

    save_wav_file(recording, fs, filename)