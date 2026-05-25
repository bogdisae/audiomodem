from rx import *
from tx import *
from helper import *

#Hello

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