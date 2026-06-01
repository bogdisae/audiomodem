import pickle
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import questionary
from pathlib import Path


def auto_correlation(x, k):
    y = []
    for i in range(0, len(x)-k):
        y.append(x[i]*x[i+k])
    return y

def save_dict(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_dict(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def record(t=3, savefile = False, filedir='bogdan/recordings/', fs=44100):
    recording = sd.rec(int(t * fs), samplerate=fs, channels=1).reshape(int(t*fs))
    sd.wait()
    if savefile:
        write(filedir+"output.wav", fs, recording) 
        np.savetxt(filedir+"output.txt", recording, delimiter="\n") 
    return recording
    
# def write_csv(filename, data, headers=None):
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         if headers:
#             writer.writerow(headers)
#         writer.writerows(data)


def synchronisation_plot(recording, correlation, windowed, h, start_index = None):
    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    ax[0, 0].plot(recording)
    ax[0, 0].set_title("a) Channel response")
    ax[0, 1].plot(correlation)
    ax[0, 1].set_title("b) Corellation: c[k] = y[x]y[x+d]")
    ax[1, 0].plot(windowed)
    ax[1, 0].set_title("c) Rectangular window correlation")
    ax[1, 1].plot(h)
    ax[1, 1].set_title("d) Estimated channel")

    if start_index is not None:
        ax[0, 0].axvline(x=start_index, color='red', linestyle='--', label='start')
        ax[0, 1].axvline(x=start_index, color='red', linestyle='--', label='start')
        ax[1, 0].axvline(x=start_index, color='red', linestyle='--', label='start')

    plt.show()

def plot_constellation(symbols):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.scatter(symbols.real, symbols.imag)
    ax.set_title("received symbols")
    plt.show()

def plot_signal(title : str, signal : np.ndarray, v_line_index, second_v_line_index=None, abs = False):
    # V_line_index draws a vertical line at an index of interest. Set to -1 to remove

    if abs : signal = np.abs(signal) # To plot absolute value

    x = np.arange(len(signal))
    plt.figure()
    plt.plot(x, signal)

    if v_line_index != -1 : plt.axvline(v_line_index, color='r')
    if second_v_line_index is not None and second_v_line_index != -1:
        plt.axvline(second_v_line_index, color='g')

    plt.title(title)
    plt.xlabel("Sample index")
    #plt.ylabel()
    plt.show()

def plot_multiple_channel_estimates(H_list):
    H_avg = np.mean(H_list, axis=0)

    H_sorted = np.sort(H_list, axis=0)
    H_trimmed = H_sorted[1:-1]   # drop min/max
    H_trimmed_avg = np.mean(H_trimmed, axis=0)

    num = len(H_list)

    plt.figure(figsize=(12, 6))

    for i in range(num):
        plt.plot(20*np.log10(np.abs(H_list[i]) + 1e-12), alpha=0.5, label=f"chirp {i}" if i < 5 else None)

    plt.plot(20*np.log10(np.abs(H_avg) + 1e-12), color='black', linewidth=2, label="AVERAGE")
    plt.plot(20*np.log10(np.abs(H_trimmed_avg) + 1e-12), color='grey', linewidth=2, label="AVERAGE - TRIMMED")

    plt.title("Channel Estimates per Chirp + Average")
    plt.xlabel("Subcarrier index")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    plt.legend()
    plt.show()



def csv_to_data_bytes(filename: str) -> np.ndarray:
    """
    Read a CSV file containing comma-separated byte values
    and return them as a NumPy uint8 array.
    """

    with open(filename, "r") as f:
        text = f.read()

    byte_list = [
        int(x.strip())
        for x in text.split(",")
        if x.strip() != ""
    ]

    return np.array(byte_list, dtype=np.uint8)


def pick_csv_file(prompt_text: str, folder: Path) -> str:
    csv_files = sorted(folder.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f'No .csv files found in {folder}')

    choice = questionary.select(
        prompt_text,
        choices=[path.name for path in csv_files],
    ).ask()

    if choice is None:
        raise SystemExit('No file selected')

    return str(folder / choice)

def pick_text_file(prompt_text: str, folder: Path) -> str:
    txt_files = sorted(folder.glob('*.txt'))
    
    if not txt_files:
        raise FileNotFoundError(f'No .txt files found in {folder}')

    choice = questionary.select(
        prompt_text,
        choices=[path.name for path in txt_files],
    ).ask()

    if choice is None:
        raise SystemExit('No file selected')

    return str(folder / choice)


def pick_wav_file(prompt_text: str, folder: Path) -> str:
    wav_files = sorted(folder.glob('*.wav'))
    if not wav_files:
        raise FileNotFoundError(f'No .wav files found in {folder}')

    choice = questionary.select(
        prompt_text,
        choices=[path.name for path in wav_files],
    ).ask()
    if choice is None:
        raise SystemExit('No file selected')
    return str(folder / choice)

def pick_m4a_file(prompt_text: str, folder: Path) -> str:
    m4a_files = sorted(folder.glob('*.m4a'))
    if not m4a_files:
        raise FileNotFoundError(f'No .m4a files found in {folder}')

    choice = questionary.select(
        prompt_text,
        choices=[path.name for path in m4a_files],
    ).ask()
    if choice is None:
        raise SystemExit('No file selected')
    return str(folder / choice)

def normalise_signal(signal):
    signal = signal.astype(np.float32)
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


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


def save_wav_file(signal, fs):
    # Also save the wav file for future use
    combined_int16 = np.int16(signal * 32767) # Convert to wav amplitudes
    filename = questionary.text("Enter filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Pipeline 2/Audio Files/{filename}.wav", fs, combined_int16)



def plot_constellation(symbols, title="Constellation Diagram", show=True):
    """
    Plots complex data symbols on the IQ plane.

    Parameters:
        symbols (np.ndarray): Array of complex symbols
        title (str): Plot title
        show (bool): Whether to call plt.show()
    """
    symbols = np.asarray(symbols)

    plt.figure(figsize=(6, 6))
    plt.scatter(symbols.real, symbols.imag, s=10)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")

    if show:
        plt.show()



#------------------------------------------------------------------------------------------------
# FUNCTIONS FOR TESTING EFFECTIVENESS

import csv

def csv_bytes_to_binary_sequence(path):

    with open(path, 'r') as f:
        data = f.read()

    decimal_bytes = [int(x) for x in data.split(',')]

    binary_sequence = []

    for byte in decimal_bytes:
        binary_sequence.extend(list(format(byte, '08b')))

    return binary_sequence

def calculate_ber(seq1, seq2):
    """
    Calculates BER between two binary sequences.

    Sequences can be:
    - lists of strings ['0','1',...]
    - lists of ints [0,1,...]
    - strings "010101"

    The longer sequence is truncated to match the shorter one.
    """

    # Convert everything to strings
    seq1 = [str(bit) for bit in seq1]
    seq2 = [str(bit) for bit in seq2]

    # Match lengths
    min_len = min(len(seq1), len(seq2))

    seq1 = seq1[:min_len]
    seq2 = seq2[:min_len]

    # Count errors
    errors = sum(b1 != b2 for b1, b2 in zip(seq1, seq2))
    if min_len != 0:
        ber = errors / min_len
    else:
        print(f'Sequence with zero length is: {1 if len(seq1) == 0 else 2}')
        raise ValueError("Sequences are empty, cannot compute BER")

    return ber, errors, min_len

def plot_pilot_phase(H1, H2, plotting_mask, section_index,f,a_meas, phase_diff):
    from matplotlib import pyplot as plt
        

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(9, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # Magnitude of channel estimate for current section
    ax1.plot(np.abs(H1*plotting_mask))
    ax1.set_title(f'H magnitude (idx {section_index})')
    ax1.set_xlabel('Frequency Bin')
    ax1.set_ylabel('Magnitude')

    # Magnitude of channel estimate for next section
    ax2.plot(np.abs(H2*plotting_mask))
    ax2.set_title(f'H magnitude (idx {section_index + 1})')
    ax2.set_xlabel('Frequency Bin')
    ax2.set_ylabel('Magnitude')

    # Phase of channel estimate for current section
    ax3.plot(np.angle(H1*plotting_mask))
    ax3.set_title(f'H phase (idx {section_index})')
    ax3.set_xlabel('Frequency Bin')
    ax3.set_ylabel('Phase (rad)')

    # Phase of channel estimate for next section
    ax4.plot(np.angle(H2*plotting_mask))
    ax4.set_title(f'H phase (idx {section_index + 1})')
    ax4.set_xlabel('Frequency Bin')
    ax4.set_ylabel('Phase (rad)')

    # Phase difference between successive channel estimates
    ax5.plot(phase_diff*plotting_mask)
    ax5.plot(f, a_meas * f *plotting_mask, 'r--', label=f'Linear fit: a={a_meas:.2e} rad/Hz')
    ax5.plot(f, a_meas * f *plotting_mask +np.pi, 'g--', label=f'Linear fit: a={a_meas:.2e} rad/Hz')
    ax5.plot(f, a_meas * f *plotting_mask -np.pi, 'g--', label=f'Linear fit: a={a_meas:.2e} rad/Hz')
    ax5.set_title(f'Phase difference (idx {section_index + 1} / idx {section_index})')
    ax5.set_xlabel('Frequency Bin')
    ax5.set_ylabel('Phase (rad)')

    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    plt.tight_layout(pad=2.0)
    plt.show()

    def gen_colour_seq(known_bit_stream, constellation):
        