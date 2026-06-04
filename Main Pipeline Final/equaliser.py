import numpy as np
from scipy.signal import chirp, correlate
import matplotlib.pyplot as plt
from helper import plot_signal, plot_multiple_channel_estimates, plot_Golay_diagnostics, estimate_delay_spread
from scipy.linalg import solve_toeplitz

class Equaliser:

    lengthInSamples : int
    lengthInSeconds : int
    preambleStartOffset : int

    def __init__(self, fs=48000, sync = False, est = False):
        self.fs = fs
        self.sync = sync
        self.est = est
        # Variable that knows where it is in the whole preamble (relative to preamble start)
        self.preambleStartOffset = None 
        

    def generate(self) -> np.ndarray: 
        raise NotImplementedError

    # FUNCTIONS TO BE IMPLEMENTED IN CHILD CLASSES
    def synchronise(self, signal: np.ndarray, plot = True):
        raise NotImplementedError

    def estimate(self, signal: np.ndarray, sync_index, plot = True):
        raise NotImplementedError


class RepeatedChirp(Equaliser):
    def __init__(self, numRepeats, chirpLength, silenceLength, f0, f1, sync = False, est = False, fs=48000):
        super().__init__(fs, sync, est)

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
            plot_signal("Transmitted key", key, -1)
            plot_signal("Received signal", signal, -1)
            plot_signal("Correlation plot", corr, key_start_index, True)

        return key_start_index
    
    def estimate(self, rxSignal: np.ndarray, sync_index, plot = True):
        
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
    def __init__(self, golay_order = 12, silence = 2048, numPairs=4, seed=(1,1), sync=False, est=False, fs=48000):
        super().__init__(fs, sync, est)

        self.golay_order = golay_order
        self.indivLength = 2**self.golay_order #Pairs are the same length - same as OFDM block length assumed (correct? - may not have to be)
        self.silence = silence #Should be longer than the channel impulse response to avoid inter-pair interference
        self.numPairs = numPairs

        self.blockLength = 2 * self.indivLength + self.silence #A, silence, B counted as a block
        self.lengthInSamples = self.silence +self.blockLength * numPairs 
        self.lengthInSeconds = self.lengthInSamples / self.fs

        self.a_ref, self.b_ref = self.generate_pair(seed) # Generate a reference pair for diagnostic plots

    def generate_pair(self, seed):

        a, b = np.atleast_1d(seed[0]), np.atleast_1d(seed[1])
        print(a,"\n",b)
        for _ in range(self.golay_order):
            a_next = np.concatenate([a, b])
            b_next =np.concatenate([a, -b])

            a, b = a_next, b_next

        assert len(a) == self.indivLength
        assert len(b) == self.indivLength
        return a, b


    def generate(self) -> np.ndarray: #Expecting np.darray
        silence_arr = np.zeros(self.silence)
        pair_sections = np.concatenate([self.a_ref, silence_arr, self.b_ref])
        pair_rep = ["A", "silence", "B"]   
        
        signal = np.concatenate([silence_arr, np.tile(pair_sections, self.numPairs)])
        signal_rep = np.concatenate([["silence"], np.tile(pair_rep, self.numPairs)])
        print(signal_rep)
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
        
        
        for i in range(self.numPairs):
            #print(f'iteration {i} out of {self.numPairs}')
            start = sync_index + i * pair_stride

            a_rx = rxSignal[start:start + indiv_len]
            b_rx = rxSignal[start + indiv_len + self.silence : start + 2 * indiv_len + self.silence]

            #print(f'a_rx length: {len(a_rx)}, b_rx length: {len(b_rx)}')
            #print(f'a seq idx: {start} to {start + indiv_len}, b seq idx: {start + indiv_len + self.silence} to {start + 2 * indiv_len + self.silence}')

            corr_a = correlate(a_rx, self.a_ref, mode='full')
            corr_b = correlate(b_rx, self.b_ref, mode='full')

            #Extract causal part starting at zero lag
            h_est = corr_a[indiv_len-1:2*indiv_len-1] + corr_b[indiv_len-1:2*indiv_len-1]

            #C_aa[n] + C_bb[n] = 2N*delta[n] -> Normalise by 2N to get actual impulse response estimate

            #Correct normaisation for scaled pairs
            h_norm = h_est / (2*self.indivLength)


            #Truncate
            #print(f'Estimated impulse response for pair {i}, h_est length: {len(h_norm)}, h_est values: {h_norm}')

            H_norm = np.fft.fft(h_norm, n=self.indivLength)

            #Alternative method - Actually works
            Y_a = np.fft.fft(a_rx, n=self.indivLength)
            Y_b = np.fft.fft(b_rx, n=self.indivLength)

            A = np.fft.fft(self.a_ref, n=self.indivLength)
            B = np.fft.fft(self.b_ref, n=self.indivLength)

            H_norm_alt = (Y_a * np.conj(A) + Y_b * np.conj(B)) / (2*self.indivLength)
            
            if i == 0 and plot:
                plot_Golay_diagnostics(h_norm, corr_a, corr_b, H_norm, H_norm_alt)
            H_list.append(H_norm_alt)




        #print(f'H: {H_list}')
        H_norm_avg = np.mean(H_list, axis=0)
        

        #print(f'H values: {np.mean(np.abs(H_est_avg))}, {np.mean(np.abs(H_est_avg))}')
        
        #Estimate delay spread
        h_norm_avg = np.fft.ifft(H_norm_avg)
        delay_spread = estimate_delay_spread(h_norm_avg, self.fs)
        print(f'Estimated delay spread: {delay_spread*1e3:.2f} milliseconds. CP time should be at least this long to avoid ISI. CP length in ms: {self.indivLength/self.fs*1e3:.2f} ms')

        return H_norm_avg
        
    def initial_SFO_estimate(self, rxSignal: np.ndarray, key_start_index : int):

        N = self.indivLength
        S = self.silence
        T = 2*N + 2*S

        # Stores the lists of samples corresponding to A and B
        A_blocks = []
        B_blocks = []

        # skip INITIAL SILENCE
        base0 = key_start_index + S

        for i in range(self.numPairs):
            base = base0 + i * T

            A_i = rxSignal[base : base + N]
            B_i = rxSignal[base + N + S : base + 2*N + S]

            A_blocks.append(A_i)
            B_blocks.append(B_i)