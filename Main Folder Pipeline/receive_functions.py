import numpy as np
from scipy.signal import correlate, chirp
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

    # Also save the wav file for future use
    combined_int16 = np.int16(audio * 32767) # Convert to wav amplitudes
    filename = questionary.text("Enter filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Folder Pipeline/Audio Files/{filename}.wav", fs, combined_int16)

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
