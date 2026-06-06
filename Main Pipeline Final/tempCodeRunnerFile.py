from rx import *
from tx import *
from helper import *

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

big_shaq_data = csv_to_data_bytes("Main Pipeline 2/Data Files/BIGSHAQ.txt")

tx = Tx(constellation, big_shaq_data, None, None, None, 2048, 4096, 10, 2_000, 12_000)
tx.encode()