import numpy as np
from scipy.signal import chirp, correlate
import matplotlib.pyplot as plt
from helper import plot_signal

class Synchroniser:
    def __init__(self, fs=48000):
        self.fs = fs

    def generate(self) -> np.ndarray: 
        raise NotImplementedError

    # FUNCTIONS TO BE IMPLEMENTED IN CHILD CLASSES
    def synchronise(self, signal: np.ndarray, plot = True):
        raise NotImplementedError



class ChirpSync(Synchroniser):
    def __init__(self, f0, f1, chirpLength, fs=48000):
        super().__init__(fs)

        self.f0 = f0
        self.f1 = f1
        self.chirpLength = chirpLength
        self.lengthInSamples = chirpLength

    def generate(self) -> np.ndarray:
        t = np.arange(self.chirpLength) / self.fs

        signal = chirp(t, f0=self.f0, f1=self.f1, t1=self.chirpLength / self.fs, method='linear')

        # normalise (important for stable correlation / channel estimation)
        m = np.max(np.abs(signal))
        return signal / m if m != 0 else signal

    def synchronise(self, signal: np.ndarray, plot=True):
        print("Synchronising using single chirp")

        key = self.generate()

        # matched filter (time-reversed correlation equivalent)
        corr = correlate(signal, key, mode='valid')
        sync_index = np.argmax(np.abs(corr))

        if plot:
            # plot_signal(corr, sync_index) ARGUEMENTS NEED TO BE UPDATED
            pass

        return sync_index


class RepeatedChirpSync(Synchroniser):
    def __init__(self, numRepeats, chirpLength, silenceLength, f0, f1, fs=48000):
        super().__init__(fs)

        self.numRepeats = numRepeats
        self.chirpLength = chirpLength
        self.silenceLength = silenceLength
        self.blockLength = chirpLength + silenceLength
        self.f0 = f0
        self.f1 = f1

        self.lengthInSamples = numRepeats * (chirpLength + silenceLength)
        self.lengthInSeconds = self.lengthInSamples / self.fs


    def generate(self):

        t = np.arange(self.chirpLength) / self.fs

        single_chirp = chirp(t, f0=self.f0, t1=self.chirpLength / self.fs, f1=self.f1, method='linear')

        silence = np.zeros(self.silenceLength)

        sections = []
        for _ in range(self.numRepeats):
            sections.append(single_chirp)
            sections.append(silence)

        signal = np.concatenate(sections)

        m = np.max(np.abs(signal))
        return signal / m if m != 0 else signal

    # override parent method
    def synchronise(self, signal: np.ndarray, plot=True):
        print("Synchronising using repeated chirp")
        print("Sync signal length", len(signal))

        key = self.generate()

        corr = correlate(signal, key, mode='valid')
        key_start_index = np.argmax(np.abs(corr))

        if plot:
        #     plot_signal("Transmitted key", key, -1)
            plot_signal("Received signal", signal, -1)
            plot_signal("Correlation plot", corr, key_start_index, True)

        return key_start_index, key_start_index + self.lengthInSamples
    
    