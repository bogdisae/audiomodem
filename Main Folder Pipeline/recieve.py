#import relevant libraries

def main(params):
    ###
    #Start recording through microphone
    #save signal in relevant folder
    #
    
if __name__ == "__main__":
    params = {
        'key_type': 'chirp',
        'repeat_key_count': 1,
        'block_length': 1024,
        'cyclic_prefix_length': 32,
        'length_of_key': 50000, # length of key 
        'fs': 44100, #Generating signal
        'fs_record': 44100, #Recording signal
        'silence_duration': 0.0,
        'record duration': 30, #Length of recording
        'signal_name': 'test_signal_01.wav',
        'recording_name': 'test_recording_01.wav'

    }
    main(params)