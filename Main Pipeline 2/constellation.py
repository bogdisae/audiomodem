import numpy as np

class Constellation:
    bits_per_symbol: int
    constellation: dict
    constellation_inequalities: dict 

    def __init__(self, bits_per_symbol, constellation, constellation_inequalities):
        self.bits_per_symbol = bits_per_symbol
        self.constellation = constellation
        self.constellation_inequalities = constellation_inequalities
    
    def bits_to_symbols(self, bits):
        if len(bits)%self.bits_per_symbol!=0:
            raise Exception(f"Bit string not divisible by {self.bits_per_symbol}")
            # Pad bits instead?
        group_bits = [tuple(bits[i:i+self.bits_per_symbol]) for i in range(0, len(bits), self.bits_per_symbol)]
        return np.array([self.constellation[b] for b in group_bits], dtype=complex) # could be made faster with numpy
    
    def symbols_to_bits(self, symbols):
        bit_list = []
        for symbol in symbols:
            bits = next((k for k, cond in self.constellation_inequalities.items() if cond(symbol)))
            bit_list.extend(bits)
        return bit_list
                    
