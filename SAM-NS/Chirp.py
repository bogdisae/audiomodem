import numpy as np
from scipy.signal import chirp
from scipy.io.wavfile import write

fs = 48000
T = 1

t = np.linspace(0, T, int(fs*T), endpoint=False)

chirp_signal = chirp(t, f0=100, f1=8000, t1=T, method='linear')

# normalize
chirp_signal = chirp_signal / np.max(np.abs(chirp_signal))

# convert to 16-bit PCM
chirp_int16 = np.int16(chirp_signal * 32767)

# write WAV file
write("SAM-NS/chirp.wav", fs, chirp_int16)