import numpy as np
from scipy.signal import correlate, chirp
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import sounddevice as sd
from pathlib import Path
from scipy.io.wavfile import write
from rx_signal import RxSignal
import questionary
import plotly.graph_objects as go




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

def demodulate_ofdm_signal(params, received_signal, sync_idx):

    #start of OFDM symbol 1
    #Assumes correlation max is at the end of the key, so the first OFDM symbol starts immediately after the key
    #Therefore sync_idx is the start of the ofdm symbol, and the key is immediately before it

    early_idx = sync_idx - params['read_prefix_early_samples']

    #No. Blocks = OFDM signal length / (block length + cyclic prefix length)

    #for in range blocks:
    #Read in block length + cyclic prefix length samples, starting from sync_idx
    '''How does reading early effect the demodulation?'''

    #Call FFT function

    #Assemble all demodulated blocks into one stream of symbols

    #FOR 

def wiener_filter_coeffs(recieved_signal, original_signal, filter_N, fs, plotting = False):

    #x_n = d_n + v_n
    
    # h = R_x^{-1} * r_xd

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

    return h_wiener
'''
def compare_wiener_length(recieved_signal, original_signal, fs):

    filter_lengths = [100, 300, 500, 800, 1024]
    filter_list = []
    freqs_list = []
    for i,filter_N in enumerate(filter_lengths):
        print(f'Calculating Wiener filter coefficients for filter length: {filter_N}')
        filter_list[i] = wiener_filter_coeffs(recieved_signal, original_signal, filter_N, fs, plotting=True)
        freqs_list[i] = np.linspace(0, fs/2, filter_N//2 + 1)


    plt.figure(figsize=(10,4))
    for i, filter_N in enumerate(filter_lengths):
        H_h = np.fft.rfft(filter_list[i])
        plt.plot(freqs_list[i], np.abs(H_h), label=f'FIR Wiener Filter Response (N={filter_N})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Wiener Filter Magnitude Response Comparison')
    plt.legend()
    plt.grid()
    plt.show()
'''

def compare_wiener_length(
    recieved_signal,
    original_signal,
    fs,
    plot_options=None
):
    if plot_options is None:
        plot_options = {
            "single_filter_response": True,
            "compare_all_filters": True,
            "show_db": True,
        }

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
    freqs = np.fft.rfftfreq(L, d=1/fs)
    '''Lines above calculate the IIR response of the Wiener filter, which is the theoretical limit of the FIR filter. We can compare this to the FIR response for different filter lengths to see how close we get to the optimal response as we increase the number of coefficients.'''


    filter_lengths = [np.nan, 100, 300, 500, 800, 1024]
    filter_list = []
    freqs_list = []

    for i, filter_N in enumerate(filter_lengths):
        if np.isnan(filter_N):
            if plot_options.get("show_db", True):
                filter_list.append(log_H_wiener)
            else:
                filter_list.append(H_wiener)
            freqs_list.append(freqs)
        else:
            if plot_options.get("show_db", True):
                filter_list.append(20 * np.log10(np.abs(filter_list[i])) + 1e-12)
            else:
                filter_list.append(wiener_filter_coeffs(recieved_signal, original_signal, filter_N, fs, plotting=False))
            
            print(f'Calculating Wiener filter coefficients for filter length: {filter_N}')
            freqs_list.append(np.linspace(0, fs / 2, filter_N // 2 + 1))

    fig = go.Figure()

    for i, filter_N in enumerate(filter_lengths):
        if np.isnan(filter_N):
            fig.add_trace(go.Scatter(
                x=freqs_list[i],
                y=np.abs(H_wiener),
                mode="lines",
                name=f"IIR Wiener Filter Response (N={filter_N})"
            ))
        else:
            H_h = np.fft.rfft(filter_list[i])
            fig.add_trace(go.Scatter(
                x=freqs_list[i],
                y=np.abs(H_h),
                mode="lines",
                name=f"FIR Wiener Filter Response (N={filter_N})"
            ))

    fig.update_layout(
        title="Wiener Filter Magnitude Response Comparison",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude"
    )

    fig.show()

    '''if plot_options.get("compare_all_filters", True):
        plt.figure(figsize=(10, 4))
        for i, filter_N in enumerate(filter_lengths):
            H_h = np.fft.rfft(filter_list[i])
            plt.plot(freqs_list[i], np.abs(H_h), label=f'FIR Wiener Filter Response (N={filter_N})')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Wiener Filter Magnitude Response Comparison')
        plt.legend()
        plt.grid()
        plt.show()

    if plot_options.get("show_db", True):
        plt.figure(figsize=(10, 4))
        for i, filter_N in enumerate(filter_lengths):
            H_h = np.fft.rfft(filter_list[i])
            plt.plot(freqs_list[i], 20 * np.log10(np.abs(H_h) + 1e-12), label=f'FIR Wiener Filter Response (N={filter_N})')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Wiener Filter Magnitude Response Comparison (dB)')
        plt.legend()
        plt.grid()
        plt.show()'''