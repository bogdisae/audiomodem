import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from pathlib import Path
import os

def generate_key(length, fs, type):
    if type == 'chirp':
        t = np.linspace(0, length/fs, length)
        f0 = 20
        f1 = 1000
        key = np.cos(2*np.pi * (f0*t + (f1-f0)*t**2/(2*length/fs)))
    
    return key.astype(np.float32)

def repeat_key_and_silence_pad(key, fs, repeat_count, silence_duration):
    # insert silence gaps between keys
    final_signal = np.empty(0, dtype=np.float32)
    key_with_silence = np.concatenate((key, np.zeros(int(silence_duration*fs), dtype=np.float32)))
    for i in range(repeat_count):
        final_signal = np.concatenate((final_signal, key_with_silence))

    #insert silence at the beginning of the signal to allow for real world operation time
    final_signal = np.insert(final_signal, 0, np.zeros(int(1*fs), dtype= np.float32))

    return final_signal

def save_wav_file(data, fs, filename):
    """Save `data` to a WAV file next to this script as PCM16.

    This avoids hard-coded/incorrect directories and converts float arrays
    into signed 16-bit PCM appropriate for `scipy.io.wavfile.write`.
    """

    # Clean filename (handle Windows/Unix paths)
    f_name = Path(filename).name

    # Ensure extension
    if not f_name.endswith('.wav'):
        f_name += '.wav'

    output_name = f_name.replace('.wav', '_Output.wav')

    # Determine output path: same folder as this script
    out_path = Path(__file__).parent / output_name

    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare PCM data
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.floating):
        # Normalize if necessary then convert to int16
        max_abs = float(np.max(np.abs(arr))) if arr.size else 1.0
        if max_abs > 1.0:
            arr = arr / max_abs
        pcm = (arr * 32767).astype(np.int16)
    else:
        pcm = arr.astype(np.int16)

    write(str(out_path), fs, pcm)

    print(f"Saved WAV file: {out_path}")

def main():

    #user params
    key_type = 'chirp'
    repeat_key_count = 5

    #length in samples, fs in Hz
    length = 5000
    fs = 44100
    key_time = length/fs
    print(f"Key duration: {key_time:.3f} seconds")

    #create desired key
      
    key = generate_key(length, fs, key_type)
    
    #Assemble multiple keys together to create a longer signal with padding
    
    test_signal = repeat_key_and_silence_pad(key, fs, repeat_count=repeat_key_count, silence_duration=0.1)

    
    print(len(test_signal))
    plt.plot(test_signal)
    plt.show()

    #save the signal as a wav file
    print("saving....")
    save_wav_file(test_signal, fs, key_type + '_' + str(repeat_key_count) + '_repeats.wav')


if __name__ == "__main__":
    main()