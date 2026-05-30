import numpy as np
from scipy.signal import chirp, correlate
import matplotlib.pyplot as plt
from helper import plot_signal, plot_multiple_channel_estimates


class Equaliser:
    def __init__(self, fs=48000):
        self.fs = fs

    def generate(self) -> np.ndarray: 
        raise NotImplementedError

    # FUNCTIONS TO BE IMPLEMENTED IN CHILD CLASSES
    def synchronise(self, signal: np.ndarray, plot = True):
        raise NotImplementedError

    def estimate(self, signal: np.ndarray, sync_index, plot = True):
        raise NotImplementedError


class Chirp(Equaliser):
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

    def estimate(self, rxSignal: np.ndarray, sync_index: int, plot = True):
        """
        Simple single-shot channel estimate using the chirp only.
        """
        key = self.generate()

        start = sync_index
        segment = rxSignal[start:start + self.chirpLength]

        # FFT-based channel estimate
        X = np.fft.fft(key, n=self.chirpLength)
        Y = np.fft.fft(segment, n=self.chirpLength)

        eps = 1e-12
        H = Y / (X + eps)

        return H


class RepeatedChirp(Equaliser):
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
    
    def estimate(self, rxSignal: np.ndarray, sync_index, plot = True):

        # Find the estimated sample index where the data starts
        dataStartIdx = sync_index + self.lengthInSamples
        
        t = np.arange(self.chirpLength) / self.fs
        singleChirp = chirp(t, f0=self.f0, t1=self.chirpLength / self.fs, f1=self.f1, method='linear')

        # FFT of known transmitted chirp
        X = np.fft.fft(singleChirp, n=self.chirpLength)
        H_list = []

        for i in range(self.numRepeats):

            start = sync_index + i * self.blockLength
            segment = rxSignal[start:start + self.chirpLength]
            
            # FFT of received chirp
            Y = np.fft.fft(segment, n = self.chirpLength)

            eps = 1e-12

            H = Y / (X + eps)

            H_list.append(H)

        H_list = np.array(H_list)

        if plot:
            plot_multiple_channel_estimates(H_list)

        # H_avg = np.mean(H_list, axis = 0)
        if len(H_list) > 2:
            H_sorted = np.sort(H_list, axis=0)
            H_trimmed = H_sorted[1:-1]
            H_avg = np.mean(H_trimmed, axis=0)
        else:
            H_avg = np.mean(H_list, axis=0)
        return H_avg
    

class GolayPairs(Equaliser):
    def __init__(self, indivLength, pairSilence, numPairs=1, fs=48000):
        super().__init__(fs)

        self.indivLength = indivLength #Pairs are the same length - same as OFDM block length assumed (correct? - may not have to be)
        self.pairSilence = pairSilence #Should be longer than the channel impulse response to avoid inter-pair interference
        self.numPairs = numPairs

        self.blockLength = 2 * indivLength + pairSilence
        self.lengthInSamples = self.blockLength * numPairs
        self.lengthInSeconds = self.lengthInSamples / self.fs

        self.a_ref, self.b_ref = self.generate_pair(seed=0) # Generate a reference pair for diagnostic plots

    def generate_pair(self, seed):
        rng = np.random.default_rng(seed)

        a = np.array([rng.choice([-1, 1])], dtype=int)
        b = np.array([a[0]], dtype=int)

        for _ in range(int(np.log2(self.indivLength))):
            a_next = np.concatenate([a, b])
            b_next = np.concatenate([a, -b])
            a, b = a_next, b_next

        assert len(a) == self.indivLength
        assert len(b) == self.indivLength
        return a, b


    def generate(self) -> np.ndarray: #Expecting np.darray
        silence = np.zeros(self.pairSilence)
        pair_sections = []

        for i in range(self.numPairs):
            pair_sections.append(self.a_ref)
            if self.pairSilence > 0:
                pair_sections.append(silence)
            pair_sections.append(self.b_ref)
            if self.numPairs > (i+1): #Only add silence between pairs, not after final pair
                '''Not sure if this silence is accounted for if multiple pairs - may need to be added'''
                pair_sections.append(silence)
        print('length of pair sections:', [len(section) for section in pair_sections])
        signal = np.concatenate(pair_sections)
        print(f'indivLength: {self.indivLength}, pairSilence: {self.pairSilence}, Length of signal: {len(signal)}')
        m = np.max(np.abs(signal))
        return signal / m if m != 0 else signal

    def synchronise(self, signal: np.ndarray, plot = True):
        raise NotImplementedError

    def estimate(self, rxSignal: np.ndarray, sync_index, pair_counter = 1, plot = True):
        
        #If pair is in middle of data -> Sync idx is the start of the pair? Would this be accurate?

        H_list = []

        indiv_len = self.indivLength
        pair_stride = self.blockLength

        self.a_ref, self.b_ref = self.generate_pair(seed=0) # Generate a reference pair for diagnostic plots
        
        
        '''CORRELATE in the iteration is not liked'''


        for i in range(self.numPairs):
            
            start = sync_index + i * pair_stride

            a_rx = rxSignal[start:start + indiv_len]
            b_rx = rxSignal[start + indiv_len + self.pairSilence : start + 2 * indiv_len + self.pairSilence]

            corr_a = correlate(a_rx, self.a_ref, mode='full')
            corr_b = correlate(b_rx, self.b_ref, mode='full')

            #Extract causal part starting at zero lag
            h_est = corr_a[indiv_len-1:2*indiv_len-1] + corr_b[indiv_len-1:2*indiv_len-1]

            #C_aa[n] + C_bb[n] = 2N*delta[n] -> Normalise by 2N to get actual impulse response estimate

            #Correct normaisation for scaled pairs
            denom = np.sum(a_rx**2) + np.sum(b_rx**2)
            h_norm = h_est / (denom) if denom != 0 else h_est / (2*self.indivLength)

            #Truncate
            print(f'Estimated impulse response for pair {i}: {h_norm}')

            H_norm = np.fft.fft(h_norm, n=self.indivLength)
            

            H_list.append(H_norm)


        #print(f'H: {H_list}')
        H_norm_avg = np.mean(H_list, axis=0)
        

        #print(f'H values: {np.mean(np.abs(H_est_avg))}, {np.mean(np.abs(H_est_avg))}')
        
        return H_norm_avg