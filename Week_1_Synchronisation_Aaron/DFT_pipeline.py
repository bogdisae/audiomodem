
import numpy as np

def iDFT_pipeline(symbols, b_len = 1024, cp_len = 32):
    
    #Chop into OFDM symbols  
    blocks = [symbols[i:i+511] for i in range(0, len(symbols), 511)]  # Reshape into blocks of b_len symbols

    print(f'Shape of Blocks: {len(blocks)}')
    iDFT_output = []
    for block in blocks:
        
        if len(block) < 511:
            block = np.pad(block, (0, 511 - len(block)))  # pad last block
        # Place modulated complex symbols from 1 block into specific frequency bins - subcarriers
        # No. frequency bins is b_len. 
        X = np.zeros((b_len), dtype=np.complex128)

        X[1:512] = np.block([block[:511]])
        #Enfore Hermiticity for real time domain signal
        X[513:] = np.conj(X[1:512][::-1])

        x = np.fft.ifft(X).real
        #Should be real from Hermitian symmetry, but take real part to avoid numerical issues

        #apply cyclic prefix
        cp = x[-cp_len:]

        x_cp = np.concatenate((cp, x))

        x_cp /= np.max(np.abs(x_cp))  # Normalize to prevent clipping

        iDFT_output.append(x_cp)
    tx_signal = np.concatenate(iDFT_output)
    return tx_signal
