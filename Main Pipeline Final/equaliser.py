from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.signal import chirp, correlate
import matplotlib.pyplot as plt
from constellation import Constellation
from helper import plot_signal, plot_multiple_channel_estimates, plot_Golay_diagnostics, estimate_delay_spread, plot_pilot_phase

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
            #plot_signal("Received signal", signal, -1)
            #plot_signal("Correlation plot", corr, key_start_index, True)
            pass

        return key_start_index
    
    def estimate(self, rxSignal: np.ndarray, sync_index, plot = True):
        
        t = np.arange(self.chirpLength) / self.fs
        singleChirp = chirp(t, f0=self.f0, t1=self.chirpLength / self.fs, f1=self.f1, method='linear')

        # FFT of known transmitted chirp
        X = np.fft.fft(singleChirp, n=self.chirpLength)
        H_list = []

        # SKIP THE FIRST CHIRP AS IT DOES NOT HAVE CYCLIC PREFIX EFFECT
        for i in range(1, self.numRepeats):

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
            pass

        # H_avg = np.mean(H_list, axis = 0)
        if len(H_list) > 2:
            H_sorted = np.sort(H_list, axis=0)
            H_trimmed = H_sorted[1:-1]
            H_avg = np.mean(H_trimmed, axis=0)
        else:
            H_avg = np.mean(H_list, axis=0)
        return H_avg
    

    def initial_SFO_estimate(self, rxSignal: np.ndarray, key_start_index : int, bin_low : int, bin_high : int, plot = True):
        
        t = np.arange(self.chirpLength) / self.fs
        singleChirp = chirp(t, f0=self.f0, t1=self.chirpLength / self.fs, f1=self.f1, method='linear')

        # FFT of known transmitted chirp
        X = np.fft.fft(singleChirp, n=self.chirpLength)

        chirp_list = []

        # SKIP THE FIRST CHIRP AS IT DOES NOT HAVE CYCLIC PREFIX EFFECT
        for i in range(1, self.numRepeats):

            start = key_start_index + i * self.blockLength
            segment = rxSignal[start:start + self.chirpLength]
            
            # FFT of received chirp
            Y = np.fft.fft(segment, n = self.chirpLength)

            chirp_list.append(Y)

        chirp_list = np.array(chirp_list)

        if plot:

            phase_accum = []

            for i in range(len(chirp_list)):
                for j in range(i + 1, len(chirp_list)):
                    ratio = chirp_list[j] / (chirp_list[i] + 1e-12)

                    ratio = ratio[bin_low:bin_high]

                    phase = np.unwrap(np.angle(ratio))
                    normalised = phase / (j - i)
                    phase_accum.append(normalised)

                    k = np.arange(bin_low, bin_high)
                    slope, intercept = np.polyfit(k, normalised, 1)
                    print(f"From {i} to {j}, slope = ", slope)

        avg_phase = np.mean(phase_accum, axis=0)

        k = np.arange(bin_low, bin_high)

        slope, intercept = np.polyfit(k, avg_phase, 1)
        best_fit = slope * k + intercept

        plt.figure(figsize=(10, 6))

        plt.plot(k, avg_phase, label="Average phase")
        plt.plot(k, best_fit, '--', label=f"Best fit (slope={slope:.3e})")

        plt.xlabel("FFT bin")
        plt.ylabel("Average phase drift per chirp")
        plt.grid(True)
        plt.legend()

        plt.show()

        print("Slope:", slope)
        return slope
                

class GolayPairs(Equaliser):
    def __init__(self, golay_order = 12, silence = 2048, numPairs=4, seed=(1,1), sync=False, est=False, fs=48000):
        super().__init__(fs, sync, est)

        self.golay_order = golay_order
        self.indivLength = 2**self.golay_order #Pairs are the same length - same as OFDM block length assumed (correct? - may not have to be)
        self.silence = silence #Should be longer than the channel impulse response to avoid inter-pair interference
        self.numPairs = numPairs

        self.blockLength = 2 * (self.indivLength + self.silence) #A, silence, B counted as a block
        self.lengthInSamples = self.silence + self.blockLength * numPairs 
        self.lengthInSeconds = self.lengthInSamples / self.fs

        self.a_ref, self.b_ref = self.generate_pair(seed) # Generate a reference pair for diagnostic plots
        self.H_list = []
        self.phase_diff = {}
        self.a_history = []

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
        pair_sections = np.concatenate([self.a_ref, silence_arr, self.b_ref, silence_arr])
        pair_rep = ["A", "silence", "B", "silence"]   
        
        signal = np.concatenate([silence_arr, np.tile(pair_sections, self.numPairs)])
        signal_rep = np.concatenate([["silence"], np.tile(pair_rep, self.numPairs)])
        print(signal_rep)
        m = np.max(np.abs(signal))
        
        return signal / m if m != 0 else signal

    def synchronise(self, signal: np.ndarray, plot = True):
        raise NotImplementedError

    def estimate(self, rxSignal: np.ndarray, sync_index, plot = True):
        
        #If pair is in middle of data -> Sync idx is the start of the pair? Would this be accurate?

        indiv_len = self.indivLength
        pair_stride = self.blockLength

        #self.a_ref, self.b_ref = self.generate_pair(seed=0) # Generate a reference pair for diagnostic plots
        
        
        for i in range(self.numPairs):
            #print(f'iteration {i} out of {self.numPairs}')
            start = sync_index + self.silence + i * pair_stride #1 silence before a

            a_rx = rxSignal[start:start + indiv_len + self.silence]
            b_rx = rxSignal[start + indiv_len + self.silence : start + 2 * indiv_len + self.silence * 2]

            #print(f'a_rx length: {len(a_rx)}, b_rx length: {len(b_rx)}')
            #print(f'a seq idx: {start} to {start + indiv_len}, b seq idx: {start + indiv_len + self.silence} to {start + 2 * indiv_len + self.silence}')

            '''Maybe following silences must be correlated?'''
            '''POTENTIAL ERROR HERE - np.zeros is wrong since already included in the rx sequences'''
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
                h_norm_alt = np.fft.ifft(H_norm_alt)
                plot_Golay_diagnostics(h_norm, h_norm_alt, corr_a, corr_b, H_norm, H_norm_alt)
            self.H_list.append(H_norm_alt)




        #print(f'H: {H_list}')
        H_norm_avg = np.mean(self.H_list, axis=0)
        

        #print(f'H values: {np.mean(np.abs(H_est_avg))}, {np.mean(np.abs(H_est_avg))}')
        
        #Estimate delay spread
        h_norm_avg = np.fft.ifft(H_norm_avg)
        delay_spread = estimate_delay_spread(h_norm_avg, self.fs)
        print(f'Estimated delay spread: {delay_spread*1e3:.2f} milliseconds. CP time should be at least this long to avoid ISI. CP length in ms: {self.indivLength/self.fs*1e3:.2f} ms')

        return H_norm_avg
        
    def initial_SFO_estimate(self, rxSignal: np.ndarray, key_start_index : int, bin_low : int, bin_high : int, plot = False):
        
        N_fft = 2**self.golay_order #Assuming Golay indiv is the same length as OFDM blocks - As of 06/06 this is true in JOSS-F
        self.active_bins = np.zeros(N_fft, dtype=bool)
        self.active_bins[bin_low:bin_high + 1] = True

        plotting_mask = self.active_bins.astype(float)  # no duplicate

        active = self.active_bins
        f_active = np.arange(N_fft)[active]  # loop-invariant, move out

        phase_accum_per_block = []
        for i, j in combinations(range(4), 2):
            self.phase_diff[i,j] = np.angle(self.H_list[j] / (self.H_list[i]+1e-12))

            y_active = self.phase_diff[i,j][active]  # fix: apply mask here

            # Outlier rejection
            median = np.median(y_active)
            std = np.std(y_active)
            mask = np.abs(y_active - median) < 2 * std

            f_fit = f_active[mask]
            y_fit = y_active[mask]

            # Linear regression on inliers
            f_mean = np.mean(f_fit)
            y_mean = np.mean(y_fit)
            a_meas = np.sum((f_fit - f_mean) * (y_fit - y_mean)) / np.sum((f_fit - f_mean)**2)
            self.a_history.append(a_meas)
        
            #radians per bin per ofdm symbol
            if plot == True:
                f = np.arange(N_fft)
                plot_pilot_phase(self.H_list[i],self.H_list[j], plotting_mask, i, j, f, a_meas, y_mean, f_mean, self.phase_diff[i,j])
                #print("Drift per frequency bin between Golay pairs (a_meas): ", a_meas/(j-i), "radians/bin. Should be close to zero for good synchronisation.")
                print("Drift per OFDM symbol: ", a_meas * N_fft / ((j - i)* (self.blockLength)), "Estimated between pairs ", i, " and ", j)
                #time_drift_per_sec = (-a_meas * N_fft / (2*np.pi)) / (self.synchroniser.fs*self.symbol_length * self.pilot_spacing)
                #print(f"Corresponds to {time_drift_per_sec:.6g} s drift at sample rate {self.synchroniser.fs} Hz.")

            phase_drift_per_block = a_meas * N_fft / ((j - i* self.blockLength))
            phase_accum_per_block.append(phase_drift_per_block)
        #calculate the average phase accumulation across all pair combinations
        #apply the correction to the rest of the data stream by rotating the OFDM symbols in the next section by the negative of the measured phase drift with interpolated in time

        phase_accum_per_block_avg = np.mean(phase_accum_per_block)
        slope = phase_accum_per_block_avg
        return slope

class WhiteNoise(Equaliser):
    def __init__(self, lengthInSamples, constellation, sync=False, est=False, fs=48000):
        super().__init__(fs, sync, est)
        self.lengthInSamples = lengthInSamples
        self.lengthInSeconds = lengthInSamples / fs
        self.constellation = constellation

    def extract_noise_stream(self):
        #Read the file named WN_symbol.txt which contains the 4096 WN.
        base = Path(__file__).parent
        with open(base / "WN_symbol.txt", "r") as f:
            noise = np.fromstring(f.read(), sep=',')

        print(f'Generated white noise of length {len(noise)} samples, duration {self.lengthInSeconds:.2f} seconds')

        self.noise_bit_stream = noise.astype(int)
        print(len(noise))
        print(self.lengthInSamples)
        assert len(noise) == self.lengthInSamples
        #No normalisation needed since 1s and 0s rn
        
    def make_OFDM_block(self, symbols):
        X = np.zeros(self.lengthInSamples, dtype=complex)

        # Positive-frequency active bins
        X[:self.lengthInSamples//2] = symbols

        # Hermitian symmetry
        X[-self.lengthInSamples//2:] = np.conj(symbols)

        ofdm_block = np.fft.ifft(X).real
        return ofdm_block

    def generate(self):
        self.extract_noise_stream()

        self.symbols = self.constellation.bits_to_symbols(self.noise_bit_stream.astype(str))

        OFDM_block = self.make_OFDM_block(self.symbols)
        return OFDM_block
