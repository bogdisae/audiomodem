import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


def record(t=3, savefile = False, filedir='bogdan/recordings/', fs=44100):
    recording = sd.rec(int(t * fs), samplerate=fs, channels=1)
    sd.wait()
    if savefile:
        write(filedir+"output.wav", fs, recording) 
        np.savetxt(filedir+"output.txt", recording, delimiter="\n") 
    return recording

recording = record(savefile=True)
a = 2