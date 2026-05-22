import numpy as np
from scipy.signal import correlate, chirp
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import sounddevice as sd
from pathlib import Path
from scipy.io.wavfile import write
from rx_signal import RxSignal
import questionary




def normalise_signal(signal):
    signal = signal.astype(np.float32)
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val

# THIS FUNCTION CAN SO FAR ONLY GENERATE TYPES OF CHIRPS. CAN ALWAYS ADD MORE 
def generate_key(fs, T, f0, f1, type_key='chirp'):
    if type_key == 'chirp' or type_key == 'up_down_chirp':
        t = np.linspace(0, T, int(fs * T), endpoint=False)

        signal = chirp(t, f0=f0, t1=T, f1=f1, method='linear' )

        if type_key == 'up_down_chirp':
            # Reverse the arguements to make a "down" chirp
            signal += chirp(t, f0=f1, t1=T, f1=f0, method='linear')

    return signal / np.max(np.abs(signal))

def key_synchronise(rxSig, fs, T, f0, f1, key_type='chirp', plot=True):

    # SIGNAL INPUT TYPE IS RxSignal (our own class)

    if key_type == 'chirp' or key_type == 'up_down_chirp':
        # IN THIS CASE DO A MATCHED FILTER
        
        signal = np.asarray(rxSig.sigArray).squeeze()
        key_sig = generate_key(fs, T, f0, f1, type_key=key_type).squeeze()

        corr = correlate(signal, key_sig, mode='valid')
        sync_index = np.argmax(np.abs(corr))
        # Plot
        if plot == True:
            # Sample indices for x-axis
            x = np.arange(len(corr))
            plt.figure()
            plt.plot(x, np.abs(corr))
            plt.title("Matched Filter Output - Absolute value")
            plt.xlabel("Sample index")
            plt.ylabel("Correlation magnitude")
            plt.show()

    return sync_index


def save_wav_file(signal, fs):
    # Also save the wav file for future use
    combined_int16 = np.int16(signal * 32767) # Convert to wav amplitudes
    filename = questionary.text("Enter filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Folder Pipeline/Audio Files/{filename}.wav", fs, combined_int16)

def record_audio(fs, channels=1):

    print("Press ENTER to start recording...")
    input() # Enter must be pressed, bit tekky from chat here

    print("Recording... press ENTER again to stop.")

    frames = []
    recording = True

    def callback(indata, frames_count, time, status):
        if recording:
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=fs,
        channels=channels,
        dtype="float32",
        callback=callback,
    )

    stream.start()

    # wait for second ENTER in main thread
    input()

    recording = False
    stream.stop()
    stream.close()

    audio = np.concatenate(frames, axis=0)

    if channels == 1:
        audio = audio.reshape(-1)

    save_wav_file(audio, fs)

    return audio

def demodulate_ofdm_signal(params, equalised_signal, data_start_idx):

    # Start of OFDM symbol 1
    # Data_start_idx is the start of the ofdm symbol, and the key is immediately before it

    early_idx = data_start_idx - params['read_prefix_early_samples']

    #No. Blocks = OFDM signal length / (block length + cyclic prefix length)

    #for in range blocks:
    #Read in block length + cyclic prefix length samples, starting from sync_idx
    '''How does reading early effect the demodulation?'''

    #Call FFT function

    #Assemble all demodulated blocks into one stream of symbols

    #FOR 

def wiener_filter_coeffs(recieved_signal, original_signal, filter_N, fs, plotting = False):
    #Remove import when not needed, do not put at top of file as it causes circular imports with channel estimation file
    from channel_estimation import quick_plot_key_comparison

    #x_n = d_n + v_n
    
    # h = R_x^{-1} * r_xd

    if plotting == True:
        '''Quickly calculate the IIR response as the theoretical limit of the FIR Wiener filter'''
        #H(w) = S_xd(w) / S_xx(w)
        #S_xd(w) = E[X(w)D*(w)]
        X_w = np.fft.fft(recieved_signal)#, n=filter_N)
        D_w = np.fft.fft(original_signal)#, n=filter_N)
        S_xd = X_w * np.conj(D_w)

        S_xx = X_w * np.conj(X_w)

        eps = 1e-12  # Prevent divide-by-zero instability
        H_wiener = S_xd / (S_xx + eps)

        L = len(H_wiener)
        #Take only the positive frequencies (non-redundant part of the spectrum)
        H_wiener = H_wiener[:L//2 + 1]

        log_H_wiener = 20 * np.log10(np.abs(H_wiener) + 1e-12)

        #12000 samples in the key, with fs = 44100 => 6000 corresponds to 22050 Hz.
        #quick_plot_key_comparison(np.abs(H_wiener), 'Wiener Filter Magnitude Response', np.angle(H_wiener), 'Wiener Filter Phase Response', x_label='Frequency', y_label='Magnitude / Phase (radians)')
        #quick_plot_key_comparison(log_H_wiener, 'Wiener Filter Magnitude Response (dB)', np.angle(H_wiener), 'Wiener Filter Phase Response (dB)', x_label='Frequency', y_label='Magnitude (dB)')

        freqs = np.fft.rfftfreq(L, d=1/fs)
    # Now calculate the FIR Wiener filter coefficients using the autocorrelation method
    r_xx = np.correlate(recieved_signal, recieved_signal, mode='full')
    M = len(recieved_signal)
    r_xx = r_xx[M-1 : M-1+ filter_N]  # Keep only lags from 0 to filter_N-1

    #Build topelitz matrix from r_xx
    R_x = toeplitz(r_xx)

    r_xd = np.correlate(recieved_signal, original_signal, mode='full')
    r_xd = r_xd[len(r_xd) // 2:len(r_xd) // 2 + filter_N]  # Keep only lags up to filter_N

    R_x_inv = np.linalg.inv(R_x + 1e-12 * np.eye(filter_N))  # Add small value to diagonal for numerical stability
    h_wiener = R_x_inv @ r_xd
    freqs_h_wiener = np.linspace(0,fs/2, filter_N//2 + 1)

    H_h = np.fft.rfft(h_wiener)
    #quick_plot_key_comparison(np.abs(h_wiener), 'Wiener Filter Coefficients', np.angle(h_wiener), 'Phase Response', x_label='Coefficient Index', y_label='Coefficient Value')

    #plot FIR and IIR response together for comparison
    plt.figure(figsize=(10,4))
    #plot abs(h_wiener) and abs(H_wiener) on same plot
    plt.plot(freqs, np.abs(H_wiener), label='IIR Wiener Filter Response')
    plt.plot(freqs_h_wiener, np.abs(H_h), label='FIR Wiener Filter Response')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Wiener Filter Magnitude Response Comparison')
    plt.legend()
    plt.grid()
    plt.show()

    #log_plot
    plt.figure(figsize=(10,4))
    plt.plot(freqs, log_H_wiener, label='IIR Wiener Filter Response (dB)')
    plt.plot(freqs_h_wiener, 20 * np.log10(np.abs(H_h) + 1e-12), label='FIR Wiener Filter Response (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Wiener Filter Magnitude Response Comparison (dB)')
    plt.legend()
    plt.grid()
    plt.show()

    print(f'R_x shape: {R_x.shape}, r_xd shape: {r_xd.shape}')

    