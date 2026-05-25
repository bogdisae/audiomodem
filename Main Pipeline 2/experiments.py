from rx import *
from tx import *
from helper import *

# chirp response
# t, ch = chirp_signal()
# res = convolve(ch, channelResponse)
# filter = correlate(ch, res, mode='full')
# channelResponse = read_csv("bogdan/impulse_response/flat_weird.csv", has_headers=False)[1]

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
# t, n_0 = white_noise(d=.3)
# taps = 1000
# n = np.zeros(len(n_0)*3)
# n[::3] = n_0
# sent = 3*np.zeros(len(n)*2 + 3*taps)
# sent[taps:taps+len(n)] = n
# sent[2*taps+len(n): 2*taps+2*len(n)] = n
# # res = convolve(sent, channelResponse)
# # res = res + np.random.randn(len(res))
# # statistics = cross_correlation(res, taps+len(n))
# # windowed = np.convolve(statistics, np.ones(len(n)))
# # windowed = windowed / len(n)
# # fig, ax = plt.subplots(2, 2, constrained_layout=True)
# # ax[0, 0].plot(sent)
# # ax[0, 0].set_title("a) Transmitted two identical pulses of noise")
# # ax[0, 1].plot(res)
# # ax[0, 1].set_title("b) Channel response")
# # ax[1, 0].plot(statistics)
# # ax[1, 0].set_title("c) Corellation: c[k] = y[x]y[x+d]")
# # ax[1, 1].plot(windowed)
# # ax[1, 1].set_title("d) Rectangular window correlation")

# # n = n / np.max(np.abs(n))
# # # convert to 16-bit PCM
# # chirp_int16 = np.int16(n * 32767)
# write('bogdan/recordings/noise_key_48.wav', 48000, sent)
# n.dump('bogdan/recordings/noise_key_48')
# plt.show()

def synchronisation_test_key():
    n_taps = 1300 #assumed
    noise_sample_len = 13230

    n, k = Tx.create_noise_key(noise_sample_len, n_taps)

    write('bogdan/recordings/synchronisation_test.wav', 44100, k)
    n.dump("bogdan/recordings/synchronisation_test_noise")

def synchronisation_test_response():
    recording = record(t=13, savefile=False)
    reference_noise = np.load("bogdan/recordings/synchronisation_test_noise", allow_pickle=True)
    correlation_dist = 14530

    receiver = Rx(None, recording, correlation_dist, 1300, reference_noise, 0, 0)
    receiver.synchronise_noise_key()
    receiver.channel_estimate()

    synchronisation_plot(recording, receiver.correlation, receiver.windowed, receiver.h, receiver.synchronisation_index)

# synchronisation_test_key()
synchronisation_test_response()