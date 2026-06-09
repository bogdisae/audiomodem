import pickle
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt

def cross_correlation(x, k):
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

# def read_csv(filename, has_headers=True):
#     with open(filename, 'r', newline='') as f:
#         reader = csv.reader(f)
#         if has_headers:
#             headers = next(reader)
#             return headers, list(reader)
#         return None, list(reader)

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

def file_to_numpy(filepath: str) -> np.ndarray:
    with open(filepath, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8)