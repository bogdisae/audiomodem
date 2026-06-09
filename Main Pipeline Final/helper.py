import pickle
from tkinter import filedialog
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import questionary
from constellation import Constellation
from pathlib import Path
import tkinter as tk


def convert_text_to_utf8_bytes():
    text_file = Path(pick_text_file("Select message file:", Path("./Main Pipeline 2/Data Files")))
    # Read the selected text file as text, then encode it to UTF-8 bytes.
    text = text_file.read_text(encoding="utf-8")
    data_bytes = text.encode("utf-8")
    csv_data = ",".join(str(byte) for byte in data_bytes)

    # Save file as a .csv of comma-separated byte values for future use.
    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    with open(f"Main Pipeline 2/Data Files/{filename}.csv", "w", encoding="utf-8", newline="") as f:
        f.write(csv_data)
    print(data_bytes[:100])

    return data_bytes


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

def plot_constellation(symbols, colour_seq=None, title="Received Constellation"):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    if colour_seq is not None:
        scatter = ax.scatter(symbols.real, symbols.imag, c=colour_seq)
    else:
        scatter = ax.scatter(symbols.real, symbols.imag)

    ax.set_title(title)
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
    H_trimmed = H_sorted[1:-1]
    H_trimmed_avg = np.mean(H_trimmed, axis=0)

    num = len(H_list)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Magnitude
    for i in range(num):
        ax1.plot(
            20 * np.log10(np.abs(H_list[i]) + 1e-12),
            alpha=0.5,
            label=f"chirp {i+1}" if i < 5 else None
        )

    ax1.plot(
        20 * np.log10(np.abs(H_avg) + 1e-12),
        color='black',
        linewidth=2,
        label="AVERAGE"
    )

    ax1.plot(
        20 * np.log10(np.abs(H_trimmed_avg) + 1e-12),
        color='grey',
        linewidth=2,
        label="AVERAGE - TRIMMED"
    )

    ax1.set_title("Channel Estimate Magnitude")
    ax1.set_xlabel("Subcarrier index")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid()
    ax1.legend()

    # Phase
    for i in range(num):
        phase = np.mod(np.angle(H_list[i]), np.pi)
        ax2.plot(
            phase,
            alpha=0.5,
            label=f"chirp {i+1}" if i < 5 else None
        )

    phase = np.mod(np.angle(H_avg), np.pi)
    ax2.plot(
        phase,
        color='black',
        linewidth=2,
        label="AVERAGE"
    )
    phase = np.mod(np.angle(H_trimmed_avg), np.pi)
    ax2.plot(
        np.angle(phase),
        color='grey',
        linewidth=2,
        label="AVERAGE - TRIMMED"
    )

    ax2.set_title("Channel Estimate Phase")
    ax2.set_xlabel("Subcarrier index")
    ax2.set_ylabel("Phase (rad)")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
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
    write(f"Main Pipeline Final/Audio Files/{filename}.wav", fs, combined_int16)



def plot_constellation_colour_seq(symbols, colour_seq_or_title=None, title="Constellation Diagram", show=True):
    """
    Flexible constellation plot. Accepts either:
      - plot_constellation(symbols, title_string)
      - plot_constellation(symbols, colour_seq, title_string)

    `colour_seq` should be a list/array of matplotlib colour specifiers
    with the same length as `symbols`.
    """
    symbols = np.asarray(symbols)

    # Determine whether second arg is title or colour sequence
    if isinstance(colour_seq_or_title, str):
        title = colour_seq_or_title
        colour_seq = None
    else:
        colour_seq = colour_seq_or_title

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6))

    if colour_seq is None:
        ax.scatter(symbols.real, symbols.imag, s=10)
    else:
        # If lengths mismatch, fall back to single-colour scatter
        if len(colour_seq) != len(symbols):
            import warnings
            warnings.warn("colour_seq length does not match symbols; ignoring colours")
            ax.scatter(symbols.real, symbols.imag, s=10)
        else:
            # Plot points grouped by colour and add legend entries
            unique_cols = []
            for c in colour_seq:
                if c not in unique_cols:
                    unique_cols.append(c)
            for c in unique_cols:
                idx = [i for i, col in enumerate(colour_seq) if col == c]
                pts = symbols[idx]
                ax.scatter(pts.real, pts.imag, c=c, s=10, label=c)
            ax.legend(title='Colour')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.grid(True)
    ax.axis("equal")

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

def plot_error_indices(error_indices, n_bits=100000, chunk_size=1000):
    error_indices = np.asarray(error_indices)
    block_size = 4096
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # --- Panel 1: Error density per chunk ---
    n_chunks = int(np.ceil(n_bits / chunk_size))
    chunk_ids = np.clip(error_indices // chunk_size, 0, n_chunks - 1)
    chunk_counts = np.bincount(chunk_ids, minlength=n_chunks)
    error_rate = chunk_counts / chunk_size * 100
    mean_ber = len(error_indices) / n_bits * 100

    axes[0].bar(np.arange(n_chunks) * chunk_size, error_rate,
                width=chunk_size, align='edge', color='steelblue')
    axes[0].axhline(mean_ber, color='tomato', linestyle='--',
                    label=f'Mean BER = {mean_ber:.1f}%')
    axes[0].set_xlabel('Bit Position')
    axes[0].set_ylabel('Error Rate (%)')
    axes[0].set_title(f'Error Density per {chunk_size}-bit Chunk')
    axes[0].set_xlim(0, n_bits)
    axes[0].legend()

    # --- Panel 2: Accumulated errors per position within 4096-block ---
    n_blocks = int(np.ceil(n_bits / block_size))
    padded_len = n_blocks * block_size

    error_array = np.zeros(padded_len, dtype=np.uint8)
    error_array[error_indices] = 1  # indices beyond n_bits remain zero (padding)

    accumulated = error_array.reshape(n_blocks, block_size).sum(axis=0)

    axes[1].bar(np.arange(block_size), accumulated, width=1, color='steelblue')
    axes[1].axhline(accumulated.mean(), color='tomato', linestyle='--',
                    label=f'Mean = {accumulated.mean():.2f} errors/position')
    axes[1].set_xlabel('Position within 4096-block')
    axes[1].set_ylabel(f'Error count ({n_blocks} blocks)')
    axes[1].set_title('Accumulated Errors by Position within 4096-block')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

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
   
    error_indices = [i for i, (b1, b2) in enumerate(zip(seq1, seq2)) if b1 != b2]
    errors = len(error_indices)

    if min_len != 0:
        ber = errors / min_len
    else:
        print(f'Sequence with zero length is: {1 if len(seq1) == 0 else 2}')
        raise ValueError("Sequences are empty, cannot compute BER")
    
    #plot_error_indices(error_indices)

    return ber, errors, min_len

def plot_pilot_phase(H1, H2, plotting_mask, i, j,f,a_meas, y_mean, f_mean, phase_diff):
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
    ax1.set_title(f'H magnitude (idx {i})')
    ax1.set_xlabel('Frequency Bin')
    ax1.set_ylabel('Magnitude')

    # Magnitude of channel estimate for next section
    ax2.plot(np.abs(H2*plotting_mask))
    ax2.set_title(f'H magnitude (idx {j})')
    ax2.set_xlabel('Frequency Bin')
    ax2.set_ylabel('Magnitude')

    # Phase of channel estimate for current section
    ax3.plot(np.angle(H1)*plotting_mask)
    ax3.set_title(f'H phase (idx {i})')
    ax3.set_xlabel('Frequency Bin')
    ax3.set_ylabel('Phase (rad)')

    # Phase of channel estimate for next section
    ax4.plot(np.angle(H2)*plotting_mask)
    ax4.set_title(f'H phase (idx {j})')
    ax4.set_xlabel('Frequency Bin')
    ax4.set_ylabel('Phase (rad)')

    # Phase difference between successive channel estimates
    ax5.plot(phase_diff*plotting_mask)
    ax5.plot(f, (a_meas * (f - f_mean) + y_mean) * plotting_mask, 'r--', label=f'Linear fit: a={a_meas:.2e} rad/Hz')
    ax5.plot(f, (a_meas * (f - f_mean) + y_mean) * plotting_mask + np.pi, 'b--', label=f'Linear fit: a={a_meas:.2e} + 2pi rad/Hz')
    ax5.plot(f, (a_meas * (f - f_mean) + y_mean) * plotting_mask - np.pi, 'b--', label=f'Linear fit: a={a_meas:.2e} - 2pi rad/Hz')
    ax5.set_title(f'Phase difference (idx {j} / idx {i})')
    ax5.set_xlabel('Frequency Bin')
    ax5.set_ylabel('Phase (rad)')

    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    plt.tight_layout(pad=2.0)
    plt.show()

def gen_colour_seq(known_bit_stream, constellation):
    """
    Generate a colour sequence for a known bit stream given a `Constellation`.

    This maps each symbol (from groups of bits) to a fixed colour. Accepts
    bit values as strings ('0'/'1') or integers (0/1). If the bit-stream
    length is not divisible by `bits_per_symbol` the final incomplete group
    is ignored.
    """
    # Normalize bits to strings '0'/'1'
    bit_strs = [str(b) for b in known_bit_stream]
    bps = constellation.bits_per_symbol

    # Truncate to whole symbols
    n_groups = len(bit_strs) // bps
    groups = [tuple(bit_strs[i * bps:(i + 1) * bps]) for i in range(n_groups)]

    # Colour map per bit-tuple (extendable)
    if bps == 1:
        colour_map = {
            ('0',): 'blue',
            ('1',): 'red',
        }
    elif bps == 2:
        colour_map = {
            ('0', '0'): 'blue',
            ('0', '1'): 'orange',
            ('1', '0'): 'green',
            ('1', '1'): 'red',
        }
    else:
        # Generic mapping: convert tuple to integer and map to a colour cycle
        base_cols = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'grey']
        colour_map = {}
        for i, g in enumerate(sorted(constellation.constellation.keys())):
            colour_map[g] = base_cols[i % len(base_cols)]

    colour_seq = [colour_map.get(g, 'grey') for g in groups]
    return colour_seq


    
def plot_Golay_diagnostics(h_norm, h_norm_alt, corr_a, corr_b, H_norm, H_norm_alt):

    #Make all just the useful bins
    H_norm = H_norm[0:len(H_norm)//2]
    H_norm_alt = H_norm_alt[0:len(H_norm_alt)//2]
    import questionary
    plot_corr = False
    #plot_corr = questionary.select("Plot correlation results for first pair? (y/n)", choices=['y', 'n']).ask()
    if plot_corr == 'y':
        
        fig_norm, ax_alt = plt.subplots()
        ax_alt.stem(h_norm, label='h_norm', basefmt=' ')
        ax_alt.set_xlabel('Lag')
        ax_alt.set_ylabel('Impulse Response')
        ax_alt.legend()
        ax_alt.set_title('Impulse response from time-domain correlation')

        fig_b, ax_blt = plt.subplots()
        ax_blt.stem(h_norm_alt, label='h_norm_alt', basefmt=' ')
        ax_blt.set_xlabel('Lag')
        ax_blt.set_ylabel('Impulse Response')
        ax_blt.legend()
        ax_blt.set_title('Impulse response from FFT method')
        plt.show()
        
        plt.plot(h_norm)
        plt.title('Combined correlation of received a and b with reference sequences')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.show()
        fig_a, ax_a = plt.subplots()
        ax_a.stem(corr_a, label='C_aa', basefmt=' ')
        ax_a.set_xlabel('Lag')
        ax_a.set_ylabel('Correlation')
        ax_a.legend()
        ax_a.set_title('Autocorrelation of a')

        fig_b, ax_b = plt.subplots()
        ax_b.stem(corr_b, label='C_bb', basefmt=' ')
        ax_b.set_xlabel('Lag')
        ax_b.set_ylabel('Correlation')
        ax_b.legend()
        ax_b.set_title('Autocorrelation of b')
        plt.show()



        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax1, ax2, ax3, ax4 = axes.flatten()
        ax1.plot(np.abs(H_norm), label='H from time-domain correlation')
        ax1.set_title('H from time-domain correlation')
        ax1.set_xlabel('Subcarrier index')
        ax1.set_ylabel('Magnitude')
        ax1.legend()

        ax2.plot(np.abs(H_norm_alt), label='H from FFT method')
        ax2.set_title('H from FFT method')
        ax2.set_xlabel('Subcarrier index')
        ax2.set_ylabel('Magnitude')
        ax2.legend()

        ax3.plot(np.angle(H_norm), label='H from time-domain correlation')
        ax3.set_title('H from time-domain correlation')
        ax3.set_xlabel('Subcarrier index')
        ax3.set_ylabel('Phase')
        ax3.legend()

        ax4.plot(np.angle(H_norm_alt), label='H from FFT method')
        ax4.set_title('H from FFT method')
        ax4.set_xlabel('Subcarrier index')
        ax4.set_ylabel('Phase')
        ax4.legend()

        plt.tight_layout(pad=2.0)
        plt.show()

def estimate_delay_spread(h_est, fs):
    # h_est is the estimated impulse response (time-domain)
    # fs is the sampling frequency

    power = np.abs(h_est)**2
    power /= np.sum(power)  # Normalize to get power distribution

    delay_indices = np.arange(len(h_est))
    mean_delay = np.sum(delay_indices * power)
    mean_delay_squared = np.sum((delay_indices**2) * power)

    rms_delay_spread = np.sqrt(mean_delay_squared - mean_delay**2)

    delay_spread_seconds = rms_delay_spread / fs
    return delay_spread_seconds


def plot_complex_arrays_separate(arrays, labels, figsize=None):
    n = len(arrays)
    fig, axes = plt.subplots(n, 2, figsize=figsize or (10, 1 * n))
    
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    row_label_x = 0.02
    for i, (arr, label) in enumerate(zip(arrays, labels)):
        magnitude_db = 20 * np.log10(np.abs(arr) + 1e-12)
        phase_deg = np.angle(arr, deg=True)

        ax_mag, ax_phase = axes[i]

        ax_mag.plot(magnitude_db)
        ax_mag.set_ylabel("dB")
        ax_mag.grid(True)

        ax_phase.plot(phase_deg, color="tab:orange")
        ax_phase.set_ylabel("degrees")
        ax_phase.grid(True)

        # row label on the left
        mid = axes[i, 0].get_position()
        row_center = (mid.y0 + mid.y1) / 2
        fig.text(row_label_x, row_center, label,
                 va="center", ha="center", rotation=90,
                 fontsize=11, fontweight="bold")

    axes[0, 0].set_title("Magnitude (dB)")
    axes[0, 1].set_title("Phase (degrees)")

    fig.subplots_adjust(left=0.1, hspace=0.4)
    
    plt.show()



def plot_constellation_colour(symbols, title="Constellation Diagram", show=True, index_colour=False):
    """
    Plots complex data symbols on the IQ plane.

    Parameters:
        symbols (np.ndarray): Array of complex symbols
        title (str): Plot title
        show (bool): Whether to call plt.show()
    """
    symbols = np.asarray(symbols)

    plt.figure(figsize=(6, 6))
    if index_colour:
        sc = plt.scatter(symbols.real, symbols.imag, c=np.arange(len(symbols)), cmap='viridis', s=3)
        plt.colorbar(sc, label='Sample index')
    else:
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


import numpy as np
import matplotlib.pyplot as plt


def qpsk_find_centres(samples, plot=True):


    samples = np.asarray(samples, dtype=np.complex128)

    """
    Find QPSK constellation centres using:
        - Quadrant mean      (green)
        - Quadrant median    (blue)
        - Robust log-mean    (red)

    Parameters
    ----------
    samples : np.ndarray
        Complex-valued QPSK samples.

    plot : bool
        If True, display a constellation plot.

    Returns
    -------
    centres_mean : np.ndarray
        Mean centre estimate for each quadrant.

    centres_median : np.ndarray
        Median centre estimate for each quadrant.

    centres_log : np.ndarray
        Log-mean centre estimate for each quadrant.
    """

    def complex_median(x):
        return np.median(x.real) + 1j * np.median(x.imag)

    def robust_log_mean(points):
        x = points.real
        y = points.imag

        x_log = np.sign(x) * np.log1p(np.abs(x))
        y_log = np.sign(y) * np.log1p(np.abs(y))

        mx = np.mean(x_log)
        my = np.mean(y_log)

        return (
            np.sign(mx) * (np.exp(np.abs(mx)) - 1)
            + 1j * np.sign(my) * (np.exp(np.abs(my)) - 1)
        )

    quadrants = [
        samples[(samples.real > 0) & (samples.imag > 0)],  # Q1
        samples[(samples.real < 0) & (samples.imag > 0)],  # Q2
        samples[(samples.real < 0) & (samples.imag < 0)],  # Q3
        samples[(samples.real > 0) & (samples.imag < 0)]   # Q4
    ]

    centres_mean = []
    centres_median = []
    centres_log = []

    for q in quadrants:

        if len(q) == 0:
            centres_mean.append(np.nan + 1j*np.nan)
            centres_median.append(np.nan + 1j*np.nan)
            centres_log.append(np.nan + 1j*np.nan)
            continue

        centres_mean.append(np.mean(q))
        centres_median.append(complex_median(q))
        centres_log.append(robust_log_mean(q))

    centres_mean = np.array(centres_mean)
    centres_median = np.array(centres_median)
    centres_log = np.array(centres_log)

    if plot:
        plt.figure(figsize=(8, 8))

        plt.scatter(
            samples.real,
            samples.imag,
            s=5,
            alpha=0.5,
            label="Samples"
        )

        plt.scatter(
            centres_mean.real,
            centres_mean.imag,
            c="green",
            marker="x",
            s=200,
            linewidths=3,
            label="Mean"
        )

        plt.scatter(
            centres_median.real,
            centres_median.imag,
            c="blue",
            marker="+",
            s=200,
            linewidths=3,
            label="Median"
        )

        plt.scatter(
            centres_log.real,
            centres_log.imag,
            c="red",
            marker="o",
            s=100,
            facecolors="none",
            linewidths=3,
            label="Log mean"
        )

        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.title("QPSK Centre Estimates")
        plt.xlabel("In-Phase")
        plt.ylabel("Quadrature")
        plt.show()

    return centres_mean, centres_median, centres_log


def save_csv_file(samples, length, filename):
    csv_data = ','.join(str(b) for b in samples[:length])

    f_name = filename.split('/')[-1]
    print(f_name)
    
    # Determine output path: same folder as this script
    out_path = Path(__file__).parent / f_name

    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        f.write(csv_data)

def file_to_numpy(filepath: str) -> np.ndarray:
    with open(filepath, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8)
    
def pick_file(prompt: str, initial_dir: Path) -> Path:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=prompt,
        initialdir=initial_dir
    )
    root.destroy()
    if not file_path:
        raise FileNotFoundError("No file selected.")
    return Path(file_path)
