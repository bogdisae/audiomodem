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

def matched_filter_plot(corr : np.ndarray, sync_index):

    x = np.arange(len(corr))
    plt.figure()
    plt.plot(x, np.abs(corr))
    plt.axvline(sync_index, color='r')
    plt.title("Matched Filter Output - Absolute value")
    plt.xlabel("Sample index")
    plt.ylabel("Correlation magnitude")
    plt.show()

def plot_multiple_channel_estimates(H_list):
    H_avg = np.mean(H_list, axis=0)

    num = len(H_list)

    plt.figure(figsize=(12, 6))

    for i in range(num):
        plt.plot(20*np.log10(np.abs(H_list[i]) + 1e-12), alpha=0.5, label=f"chirp {i}" if i < 5 else None)

    plt.plot(20*np.log10(np.abs(H_avg) + 1e-12), color='black', linewidth=2, label="AVERAGE")

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