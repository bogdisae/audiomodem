from rx import *
from tx import *
from helper import *
from channel import *
from scipy.io.wavfile import write, read

#Hello

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

repeated_chirp_equaliser = RepeatedChirp(2, 1024, 1024, 0, 20000)

big_shaq_data = csv_to_data_bytes("Main Pipeline 2/Data Files/BIGSHAQ.txt")

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

# ofdm_test()

def tx_test():
    tx = Tx(constellation, big_shaq_data, repeated_chirp_equaliser, 1024, 1024)
    tx.encode()
    write("Main Pipeline 2/Audio Files/double_chirp_gap.wav", 48000, tx.transmitted_signal)

def rx_test():
    # received = read("Main Pipeline 2/Audio Files/repeated_chirp_2_test.wav")[1]
    received = read("Main Pipeline 2/Audio Files/double_chirp_gap_output.wav")[1]
    # received = record(6,savefile=True, filedir="Main Pipeline 2/Audio Files/double_chirp_gap", fs=48000)
    rx = Rx(constellation, received, 1024, 1024, repeated_chirp_equaliser, 0)
    rx.decode()

    plot_constellation(np.array(rx.data_symbols[:100]))
    print(rx.data_bytes)

# tx_test()
# rx_test()

# Channel model

def simulated_channel():
    tx = Tx(constellation, big_shaq_data, repeated_chirp_equaliser, 1024, 1024)
    tx.encode()

    channel = Channel(tx.transmitted_signal)
    # h = np.sinc(np.linspace(0, 6*np.pi, 2000))
    h = [1]
    received = channel.apply_default_channel(h,r=.1,sfo_ppm=10)

    rx = Rx(constellation, received, 1024, 1024, repeated_chirp_equaliser, 17)
    rx.decode()

    plot_constellation(np.array(rx.data_symbols[:1024*4]), index_colour=True)
# simulated_channel()


def simulated_channel_phase_offset():
    tx = Tx(constellation, big_shaq_data, repeated_chirp_equaliser, 1024, 1024)
    tx.encode()

    channel = Channel(tx.transmitted_signal)
    # h = np.sinc(np.linspace(0, 6*np.pi, 2000))
    h = [1]
    received = channel.apply_default_channel(h,r=0,sfo_ppm=0)

    rx = Rx(constellation, received, 1024, 1024, repeated_chirp_equaliser, 0)
    rx.decode()

    symbol_offset_calc(constellation, rx.data_symbols)

    # plot_constellation(np.array(rx.data_symbols[:1024*4]), index_colour=True)

def symbol_offset_calc(C: Constellation, symbols):
    # for each symbol:
    #     offset of that symbol from the group expected?
    decoded_symbols = C.bits_to_symbols(C.symbols_to_bits(symbols))
    phase_offset = np.angle(np.conj(symbols)*decoded_symbols)
    # plot_constellation_with_second(symbols, phase_offset, "Constellation diagram with phase offset", index_colour=True)
    plot_constellation(symbols,index_colour=True)

simulated_channel_phase_offset()