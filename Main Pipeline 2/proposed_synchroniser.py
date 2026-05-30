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

        corr = np.abs(correlate(signal, key, mode='valid'))
        key_start_index = np.argmax(np.abs(corr))

        #Find second peak idx for coarse CFO estimation
        # Mask out a window around it
        
        exclusion = 200  # samples either side — adjust to be wider than your peak
        masked = corr.copy()
        masked[ : key_start_index + exclusion] = 0

        # Second peak
        second_idx = np.argmax(masked)
        if plot:
        #     plot_signal("Transmitted key", key, -1)
            plot_signal("Received signal", signal, -1)
            plot_signal("Correlation plot", corr, key_start_index, second_idx, True)
        
        return key_start_index, key_start_index + self.lengthInSamples, second_idx

    def Coarse_CFO_correction(self, signal: np.ndarray, sync_index, sync_2nd_peak):
        from scipy.signal import hilbert
        first_chirp = signal[sync_index:sync_index + self.chirpLength]
        second_chirp = signal[sync_2nd_peak:sync_2nd_peak + self.chirpLength]

        #Comlpex digital key - must be complex for phase estimation
        key = hilbert(self.generate())

        corr = correlate(signal, key, mode='valid')

        peak_one_complex = corr[sync_index]
        peak_two_complex = corr[sync_2nd_peak]

        #phi
        phase_diff = np.angle(peak_one_complex) - np.angle(peak_two_complex)

        #Time separation
        T_c = (sync_2nd_peak - sync_index) / self.fs

        #Phase rotation (phi) of arguement j 2pi delta_f T_c
        delta_f = phase_diff / (2 * np.pi * T_c)

        '''print(f"First chirp segment: {first_chirp[:10]},\nSecond chirp segment: {second_chirp[:10]}")
        print(f"Estimated phase difference between chirps: {phase_diff} radians")'''
        print(f"Estimated CFO: {delta_f} Hz")

        correction_wave = np.exp(-1j * 2 * np.pi * delta_f * np.arange(len(signal)) / self.fs)
        corrected_signal = signal * correction_wave
        return corrected_signal
    
    