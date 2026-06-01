print("Importing modules...")
import os


from proposed_rx import *
from proposed_tx import *
from helper import *
from pathlib import Path
import numpy as np
import questionary
from scipy.io import wavfile
from scipy.io.wavfile import read, write
from constellation import Constellation
from equaliser import *
from proposed_synchroniser import *
print("Modules imported successfully")

sampleRate = 48000

'''constellation = Constellation(2, {
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
'''

constellation = Constellation(1, {
    ('0',): 1,
    ('1',): -1
}, {
    ('0',): lambda s: s.real >= 0,
    ('1',): lambda s: s.real < 0
})
def m4a_to_wav():
    selected_path = pick_m4a_file("Select an M4A file:", Path("./Main Pipeline 2/Audio Files/Aaron_Recordings/Phone_rec"))
    # Use ffmpeg to convert the selected M4A file to WAV format
    selected_path = Path(selected_path)
    output_path = selected_path.parent.parent / f"{selected_path.stem}.wav"
    command = f'ffmpeg -i "{selected_path}" -ar {sampleRate} -ac 1 "{output_path}"'
    os.system(command)
    print(f"Converted {selected_path} to {output_path}")

def convert_text_to_utf8_bytes():
    text_file = Path(pick_text_file("Select message file:", Path("./Main Pipeline 2/Data Files")))
    # Read the selected text file as text, then encode it to UTF-8 bytes.
    text = text_file.read_text(encoding="utf-8")
    data_bytes = text.encode("utf-8")
    csv_data = ",".join(str(byte) for byte in data_bytes)

    # Save file as a .csv of comma-separated byte values for future use.
    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    with open(f"Main Pipeline 2/Data Files/{filename}.csv", "w", encoding="utf-8", newline="") as f:
        f.write(csv_data)
    print(data_bytes[:100])

def roundtrip_test(transmitter):
    write("tmp_tx.wav", sampleRate, np.int16(transmitter.transmitted_signal * 32767))
    fs_rx, rx_sig = read("tmp_tx.wav")
    rx_sig = rx_sig.astype(float) / 32767.0
    print("Roundtrip: wrote tmp_tx.wav and read back", fs_rx, "samples:", len(rx_sig))
    assert fs_rx == sampleRate, "Sample rate mismatch in roundtrip test"
    assert len(rx_sig) == len(transmitter.transmitted_signal), "Sample length mismatch in roundtrip test"
    print("Roundtrip test passed")

def pilot_alignment_CPE_estimation(golayPairs, synchroniser, rx_sig):
    # tx_pilot from the equaliser used by transmitter (at TX time)
    tx_pilot = golayPairs.generate()            # same object used to build Tx

    # find pilot in received signal via synchroniser
    pilot_start_idx = synchroniser.synchronise(rx_sig, False)[1] + golayPairs.pairSilence  # for RepeatedChirpSync

    rx_pilot = rx_sig[pilot_start_idx:pilot_start_idx + golayPairs.lengthInSamples]
    print(f'tx_pilot length: {len(tx_pilot)}, rx_pilot length: {len(rx_pilot)}')

   
    cpe = np.angle(np.vdot(rx_pilot, np.conj(tx_pilot)))
    print("Pilot start:", pilot_start_idx, "Pilot CPE (rad):", cpe)

def generateChirp_plus_data(standard = True):
    text_file = pick_csv_file("Select message file:", Path("./Main Pipeline 2/Data Files"))
    data_bytes = csv_to_data_bytes(text_file)


    #STANDARD CHIRP PARAMETERS
    if standard == True:
        repeatedChirp = RepeatedChirpSync(10, 1024, 1024, 20, 20000, sampleRate)
        key = repeatedChirp.generate()
        golayPairs = GolayPairs(1024, 10240, numPairs=1, fs=sampleRate)
        pilot_seq = golayPairs.generate()
        transmitter = Tx(
            constellation=constellation,
            data_bytes=data_bytes,
            equaliser=golayPairs,
            synchroniser=repeatedChirp,
            cp_length=1024,
            block_length=1024,
            pilot_spacing=10,
        )
    else:
        #Experimental CHIRP PARAMETERS
        repeatedChirp = RepeatedChirpSync(2, 1024, 1024, 0, 20000, sampleRate)
        key = repeatedChirp.generate()
        golayPairs = GolayPairs(1024, 10240, numPairs=1, fs=sampleRate)
        pilot_seq = golayPairs.generate()
        transmitter = Tx(
            constellation=constellation,
            data_bytes=data_bytes,
            equaliser=golayPairs,
            synchroniser=repeatedChirp,
            cp_length=1024,
            block_length=1024,
            pilot_spacing=10,
        )

    transmitter.encode()

    roundtrip_test(transmitter)
    
    #Plot shows all are correct
    plot_constellation(transmitter.data_symbols[-2000:], "Transmitted Constellation")
    
    #bad_mask = ~np.isfinite(transmitter.data_symbols) | np.isnan(transmitter.data_symbols)
    #print(f'Number of bad symbols: {np.sum(bad_mask)}')
    #print(f'Indices of bad symbols: {np.where(bad_mask)[0]}')
    
    sig = transmitter.transmitted_signal

    combined_int16 = np.int16(sig * 32767) # Convert to wav amplitudes
    filename = questionary.text("Enter output filename (without extension):").ask()
    if filename is None:
        raise SystemExit("No filename provided")
    write(f"Main Pipeline 2/Audio Files/Aaron_Recordings/{filename}.wav", sampleRate, combined_int16)
    print(f'Saved in dir: Main Pipeline 2/Audio Files/Aaron_Recordings/{filename}.wav')

def receiveRepeated_chirp_plus_data(standard = True):

    mode = questionary.select("Do you want to record audio or select an existing file?",
        choices=["Record audio", "Select file"]
    ).ask()

    if mode is None: raise SystemExit("No option selected")

    if mode == "Select file":
        selected_path = pick_wav_file("Select a WAV file:", Path("./Main Pipeline 2/Audio Files/Aaron_Recordings/"))
        fs_rx, sig = wavfile.read(selected_path)
        sig = normalise_signal(sig)

    elif mode == "Record audio":
        print("Recording mode selected")
        sig = record_audio(sampleRate)
        sig = normalise_signal(sig)
    
    if standard == True:
        repeatedChirp = RepeatedChirpSync(10, 1024, 1024, 20, 20000, sampleRate)
        golayPairs = GolayPairs(1024, 10240, numPairs=1, fs=sampleRate)
        receiver = Rx(constellation, sig, 1024, 1024, golayPairs, repeatedChirp, "Golay", "Block")
    else:
        repeatedChirp = RepeatedChirpSync(2, 1024, 1024, 0, 20000, sampleRate)
        golayPairs = GolayPairs(1024, 10240, numPairs=1, fs=sampleRate)
        receiver = Rx(constellation, sig, 128, 1024, golayPairs, repeatedChirp, "Golay", "Block")

    pilot_alignment_CPE_estimation(golayPairs, repeatedChirp, sig)

    receiver.decode()

    print(f'Symbols decoded: {receiver.data_symbols[:100]}')
    #print ("Number of coefficients:", len(receiver.H))
    #print("First 10 estimated coefficients:\n", receiver.H[:10])

    print(receiver.data_bits[:100])
    plot_constellation(receiver.data_symbols[0:1000])

    print('__________________________________________________________\n_____________________________________________________________________')

    text_file = pick_csv_file("Select message file:", Path("./Main Pipeline 2/Data Files"))
    data_bytes = csv_to_data_bytes(text_file)
    repeatedChirp = RepeatedChirpSync(10, 1024, 1024, 20, 20000, sampleRate)
    key = repeatedChirp.generate()
    golayPairs = GolayPairs(1024, 1024, numPairs=1, fs=sampleRate)
    pilot_seq = golayPairs.generate()
    transmitter = Tx(
            constellation=constellation,
            data_bytes=data_bytes,
            equaliser=golayPairs,
            synchroniser=repeatedChirp,
            cp_length=1024,
            block_length=1024,
            pilot_spacing=10,
        )
    transmitter.encode()

    X_tx = transmitter.data_symbols
    X_eq = receiver.data_symbols

    pilot_sym = golayPairs.generate()


    print("mean power TX (data):", np.mean(np.abs(X_tx)**2))
    print("mean power RX equalised:", np.mean(np.abs(X_eq)**2))
    #print("H mean abs:", np.mean(np.abs(H)), "min/max:", np.min(np.abs(H)), np.max(np.abs(H)))
    # verify fft/ifft identity
    diff = np.max(np.abs(np.fft.fft(np.fft.ifft(X_tx)) - X_tx))
    print("fft(ifft) identity max error:", diff)
    # pilot energy
    if 'pilot_sym' in globals():
        print("pilot mean power:", np.mean(np.abs(pilot_sym)**2))

    text_file = pick_csv_file("Select message file:", Path("./Main Pipeline 2/Data Files"))
    known_bit_seq = csv_bytes_to_binary_sequence(text_file)

    bit_check = np.linspace(0, len(known_bit_seq)-1, 5000, dtype=int)
    ber, errors, min_len = calculate_ber(known_bit_seq, receiver.data_bits[:5000])

    print("BER:", ber)
    print("Errors:", errors)
    print("Min Len", min_len)

    blocks_ber = []
    bits_per_symbol = constellation.bits_per_symbol
    bits_per_block = len(receiver.active_bins) * bits_per_symbol
    print(f"No. Blocks expected: {len(known_bit_seq) / bits_per_block}")

    for i in range(len(receiver.ofdm_blocks)):
        
        start = i * bits_per_block
        end = (i + 1) * bits_per_block
        try:
            ber_i, errors_i, min_len_i = calculate_ber(
                known_bit_seq[start:end],
                receiver.data_bits[start:end]
            )
            blocks_ber.append(ber_i)
        except:
            print(f'Disregarding final bits in partial block - Partial block BER not supported')
    
    print(f"BER variance: {np.var(blocks_ber)}")
    print(f"BER trend: {np.polyfit(range(len(blocks_ber)), blocks_ber, 1)}")

    plt.plot(blocks_ber)
    plt.title("BER per OFDM block")
    plt.show()

    print("H shape:", receiver.H.shape)
    print("H stats: min,max,mean:", np.min(np.abs(receiver.H)), np.max(np.abs(receiver.H)), np.mean(np.abs(receiver.H)))
    print("NaN/Inf in H:", np.sum(np.isnan(receiver.H)), np.sum(np.isinf(receiver.H)))

print("Functions compiled successfully")

def main():
    mode = questionary.select("Which function do you want to run?", choices=[
        "Convert M4A to WAV and run","Laptop rec and run",'Create and save']).ask()

    if mode == "Convert M4A to WAV and run":
        m4a_to_wav()
        receiveRepeated_chirp_plus_data()
    elif mode == "Laptop rec and run":
        receiveRepeated_chirp_plus_data()
    else:
        generateChirp_plus_data()

main()


################
#USE FOR REPORT
#def test_Golay():