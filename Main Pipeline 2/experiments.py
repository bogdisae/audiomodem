from rx import *
from tx import *
from helper import *
from scipy.io.wavfile import write, read

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
# synchronisation_test_response()

def ofdm_test():
    constellation = Constellation(2, {
        ('0', '0'): (1+1j)/np.sqrt(2),
        ('0', '1'): (-1+1j)/np.sqrt(2),
        ('1', '0'): (1-1j)/np.sqrt(2),
        ('1', '1'): (-1-1j)/np.sqrt(2)
    }, {
        ('0', '0'): lambda s: (s.real >= 0) & (s.imag >= 0),
        ('0', '1'): lambda s: (s.real < 0) & (s.imag >=  0),
        ('1', '0'): lambda s: (s.real >=  0) & (s.imag < 0),
        ('1', '1'): lambda s: (s.real <  0) & (s.imag <  0),
    })
    data = np.array([100, 50, 32, 42, 56, 200, 43, 53, 23, 12, 50, 32], dtype=np.uint8)
    tx = Tx(constellation, data, None, 32, 1024)
    tx.encode_symbols()
    tx.prep_ofdm_blocks()
    ofdm_blocks = np.concatenate(tx.ofdm_symbol_blocks)

    rx = Rx(constellation, ofdm_blocks / max(abs(ofdm_blocks)), 32, 1024)
    rx.synchronisation_index = 0
    rx.H = np.ones(1024)
    rx.decode()
    print(rx.data_bytes)

ofdm_test()