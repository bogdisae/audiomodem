import numpy as np
import sounddevice as sd
from scipy.signal import chirp
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import convolve, correlate
from scipy.io.wavfile import write
import csv

def chirp_signal(d = .3, f0 = 2, f1 =8000,savefile = False, fieldir="./bogdan/recordings" , fs=44100):
    t = np.linspace(0, d, int(fs * d), endpoint=False)
    signal = chirp(t, f0=f0, f1=f1, t1=d, method="linear")
    return t, signal

def white_noise(d=.03, fs=44100):
    noise = np.random.uniform(-1.0, 1.0, int(fs * d))
    t = np.linspace(0, d, int(fs * d), endpoint=False)
    return t, noise

channelResponse = []
with open("bogdan/impulse_response/flat_weird.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        # Check row exists and first column is not empty
        if row and row[0].strip() != "":
            channelResponse.append(float(row[0])) # Convert all data to floats to avoid weird errors

def channel_estimation(known, received):
    '''Assuming synchronisation, use the known sent signal and the received signal to estimate impulse response'''
    fftlen = len(received)
    Y = np.fft.fft(received, n = fftlen)
    X = np.fft.fft(known, n = fftlen)
    h = np.fft.ifft(Y/X)
    return h

def cross_correlation(x, k):
    y = []
    for i in range(0, len(x)-k):
        y.append(x[i]*x[i+k])
    return y

# chirp response
# t, ch = chirp_signal()
# res = convolve(ch, channelResponse)
# filter = correlate(ch, res, mode='full')

# fig, ax = plt.subplots(1, 3)
# ax[0].plot(ch)
# ax[1].plot(res)
# ax[2].plot(filter)
# plt.show()


# Chirp channel estimation
# t, ch = chirp_signal()
# res = convolve(ch, channelResponse)

# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(ch)
# ax[0, 1].plot(res)
# ax[1, 0].plot(channelResponse)
# ax[1, 1].plot(channel_estimation(ch, res))
# plt.show()


# White noise estimation
# t, n = white_noise()
# res = convolve(n, channelResponse)

# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(n)
# ax[0, 1].plot(res)
# ax[1, 0].plot(channelResponse)
# ax[1, 1].plot(channel_estimation(n, res))
# plt.show()

# w = 200
# d = 100
# sin = np.sin(np.linspace(0, w*0.734, w))

# # sig = np.zeros(d+w*4)
# # sig[d:w+d] = sin
# # sig[d+2*w:d+3*w] = sin
# sig = sin

# res = convolve(sig, channelResponse)
# # filter = correlate(sig, res, mode='full')

# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(sig)
# ax[0, 1].plot(channelResponse)
# ax[1, 0].plot(res)

# plt.show()


# Cross-correlation based detection
# t, n = white_noise()
# sent = np.zeros(len(n)*2 + 3*300)
# sent[300:300+len(n)] = n
# sent[600+len(n): 600+2*len(n)] = n
# res = convolve(sent, channelResponse)
# statistics = cross_correlation(res, 300+len(n))
# statistics2 = cross_correlation(sent, 300+len(n))
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(sent)
# ax[0, 1].plot(res)
# ax[1, 0].plot(statistics)
# ax[1, 1].plot(statistics2)
# plt.show()

# Cross-correlation with noise
t, n = white_noise()
sent = 3*np.zeros(len(n)*2 + 3*300)
sent[300:300+len(n)] = n
sent[600+len(n): 600+2*len(n)] = n
res = convolve(sent, channelResponse)
res = res + np.random.randn(len(res))
statistics = cross_correlation(res, 300+len(n))
windowed = np.convolve(statistics, np.ones(len(n)))
windowed = windowed / len(n)
fig, ax = plt.subplots(2, 2, constrained_layout=True)
ax[0, 0].plot(sent)
ax[0, 0].set_title("a) Transmitted two identical pulses of noise")
ax[0, 1].plot(res)
ax[0, 1].set_title("b) Channel response")
ax[1, 0].plot(statistics)
ax[1, 0].set_title("c) Corellation: c[k] = y[x]y[x+d]")
ax[1, 1].plot(windowed)
ax[1, 1].set_title("d) Rectangular window correlation")

n = n / np.max(np.abs(n))
# convert to 16-bit PCM
chirp_int16 = np.int16(n * 32767)
write('bogdan/recordings/noise.wav', 44100, np.abs(n))
plt.show()
