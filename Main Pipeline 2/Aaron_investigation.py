from rx import *
from tx import *
from helper import *

def create_file():
    #Test params
    key_len_sec = 0.5
    f0 = 0
    f1 = 24000

    key = Tx.chirp_signal(d=key_len_sec, f0=f0, f1=f1, savefile = True, fieldir="./Audio Files/Aaron_audio", fs = 48000)
    write('Main Pipeline 2/Audio_Files/Aaron_audio/chirp_key.wav', 48000, key)
    #n.dump('Audio Files/Aaron_audio/chirp_key_noise')

create_file()