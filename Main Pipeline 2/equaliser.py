import numpy as np
from scipy.signal import chirp, correlate
import matplotlib.pyplot as plt


class Equaliser:
    def __init__(self, fs=48000):
        self.fs = fs

    # FUNCTIONS TO BE IMPLEMENTED IN CHILD CLASSES
    def synchronise(self, signal: np.ndarray, plot=True) -> int:
        raise NotImplementedError

    def estimate(self, signal: np.ndarray):
        raise NotImplementedError


# class Chirp(Equaliser):
#     def __init__(self, f0, f1, fs=48000):
#         super().__init__(fs)
#         self.f0 = f0
#         self.f1 = f1

#     def synchronise(self, signal: np.ndarray, plot=True):
#         raise NotImplementedError  


class RepeatedChirp(Equaliser):
    def __init__(self, numRepeats, chirpLength, silenceLength, f0, f1, fs=48000):
        super().__init__(fs)

        self.numRepeats = numRepeats
        self.chirpLength = chirpLength
        self.silenceLength = silenceLength
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
    def synchronise(self, signal: np.ndarray, plot=True) -> int:
        print("Synchronising using repeated chirp")

        key = self.generate()

        corr = correlate(signal, key, mode='valid')
        sync_index = np.argmax(np.abs(corr))

        if plot:
            x = np.arange(len(corr))
            plt.figure()
            plt.plot(x, np.abs(corr))
            plt.axvline(sync_index, color='r')
            plt.title("Matched Filter Output - Absolute value")
            plt.xlabel("Sample index")
            plt.ylabel("Correlation magnitude")
            plt.show()

        return sync_index