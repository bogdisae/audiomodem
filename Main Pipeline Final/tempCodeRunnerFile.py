from rx import *
from tx import *
from helper import *

sampleRate = 48_000
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

repeatedChirp = RepeatedChirp(10, 4096, 0, 750, 18000, sync = True, est = True, fs = sampleRate)
golayPairs = GolayPairs(12, silence = 2048, numPairs=4, seed = (1,1), est = True, fs = sampleRate) #2**12 = 4096
whiteNoise = WhiteNoise(4096, 2048, constellation, sync = False, est = False, fs = sampleRate)

big_shaq_data = csv_to_data_bytes("Main Pipeline 2/Data Files/BIGSHAQ.txt")

tx = Tx("filename.txt", constellation, big_shaq_data, repeatedChirp, golayPairs, whiteNoise, 2048, 4096, 10, 2_000, 12_000)
tx.encode()
symbols = tx.ofdm_symbol_blocks
rx = Rx(constellation, tx.transmitted_signal, 2048, 4096, [repeatedChirp, golayPairs], None)
rx.decode()
rx.H = np.ones(4096)
rx.early_samples = 0
rx.sfo_rad_per_index_per_block = 0
rx.preamble_total_length = 0
# rx.ofdm_blocks = np.concatenate(symbols)
# rx.extract_ofdm_blocks()

# for i in range(854):
#     t = tx.ofdm_symbol_blocks[5][i]
#     r = rx.ofdm_blocks[5, i]
#     print(f"{'match' if np.isclose(t, r) else 'error'} {t.real:.5f}+{t.imag:.5f} : {r.real:.5f}+{r.imag:.5f}")

# for i in range(2000, 2500):
#     t = tx.interleaved_blocks[i//854][i%854]
#     r = rx.decoded_symbols[i]
#     print(f"{'match' if np.isclose(t, r) else 'error'} {t.real:.5f}+{t.imag:.5f} : {r.real:.5f}+{r.imag:.5f}")


for i in range(0, 300):
    t = tx.data_symbols[i]
    r = rx.decoded_symbols[i]
    print(f"{'match' if np.isclose(t, r) else 'error'} {t.real:.1f}+{t.imag:.1f} : {r.real:.1f}+{r.imag:.1f}")

#interleave
bits = tx.ldpc_bits if tx.use_ldpc else tx.data_bits
data_symbols = tx.constellation.bits_to_symbols(bits)
symbols_per_block = 854

padding_symbols = np.array(constellation.bits_to_symbols(('0', '0')))

# we map 35 ldpc blocks to 30 ofdm blocks
if (symbols_per_block != 854): raise Exception("Standard requres 854 active bins")
remainder = len(data_symbols) % (tx.c.K*35)
pad_length = 35*tx.c.K - remainder if remainder != 0 else 0

if pad_length > 0:
    padding = np.resize(padding_symbols, pad_length)
    data_symbols = np.concatenate([data_symbols, padding])

if tx.use_ldpc:
    ldpc_blocks =data_symbols.reshape(-1, 35*tx.c.K)
    interleaved_blocks = np.array([], dtype=complex)
    ldpc_skip_factor = 15839
    thirty_ofdm_block_length = 25620 # 30x854
    for ldpc_block in ldpc_blocks:
        interleaved_block = np.zeros(thirty_ofdm_block_length, dtype=complex)
        for i in range(len(ldpc_block)):
            interleaved_block[(i*ldpc_skip_factor)%thirty_ofdm_block_length]= ldpc_block[i]
        interleaved_blocks=np.concatenate([interleaved_blocks ,interleaved_block])
    blocks = np.array(interleaved_blocks).reshape(-1, symbols_per_block)
    interleaved_blocks = blocks
else:
    blocks = data_symbols.reshape(-1, symbols_per_block)

ofdm_symbol_blocks = [
    tx.prep_ofdm_block(block)
    for block in blocks
]

# Receiver
ofdm_symbol_length = rx.block_length + rx.cp_length
ofdm_blocks = np.concatenate(ofdm_symbol_blocks)
        
if rx.use_ldpc:
    remainder = len(ofdm_blocks) % (ofdm_symbol_length*30)
    ofdm_blocks = ofdm_blocks[:-remainder] if remainder > 0 else ofdm_blocks
else:
    remainder = len(ofdm_blocks) % ofdm_symbol_length
    pad_length = ofdm_symbol_length - remainder if remainder != 0 else 0
    padding_symbols = np.array(constellation.bits_to_symbols(('0', '0')))

    if pad_length > 0:
        padding = np.resize(padding_symbols, pad_length)
        ofdm_blocks = np.concatenate([ofdm_blocks, padding])

ofdm_blocks = ofdm_blocks.reshape(-1, ofdm_symbol_length)

decoded_symbols = []
for idx, ofdm_block in enumerate(ofdm_blocks):
    decoded_symbols.extend(rx.decode_ofdm_block(ofdm_block, idx))
decoded_symbols = decoded_symbols

for i in range(0, 300):
    t = blocks.flatten()[i]
    r = decoded_symbols[i]
    print(f"{'match' if np.isclose(t, r) else 'error'} {t.real:.1f}+{t.imag:.1f} : {r.real:.1f}+{r.imag:.1f}")


if rx.use_ldpc:
    deinterleaved_symbols = []
    thirty_ofdm_block_length = 25620 # 30x854
    ldpc_skip_factor = 15839
    grouped_ofdm_blocks = np.array(decoded_symbols).reshape(-1, thirty_ofdm_block_length)
    for interleaved_block in grouped_ofdm_blocks:
        ldpc_block = np.zeros(len(interleaved_block), dtype=complex)
        for i in range(len(interleaved_block)):
            print(interleaved_block[(i*ldpc_skip_factor)%thirty_ofdm_block_length])
            ldpc_block[i]=interleaved_block[(i*ldpc_skip_factor)%thirty_ofdm_block_length]
        deinterleaved_symbols.extend(ldpc_block)
    data_symbols_received = deinterleaved_symbols
else:
    data_symbols_receivec = decoded_symbols

print("Constellations")
for i in range(0, 300):
    t = data_symbols[i]
    r = data_symbols_received[i]
    print(f"{'match' if np.isclose(t, r) else 'error'} {t.real:.1f}+{t.imag:.1f} : {r.real:.1f}+{r.imag:.1f}")
