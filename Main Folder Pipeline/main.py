import questionary
import functions

def pick_wav_file(prompt_text: str, folder: Path) -> str:
    wav_files = sorted(folder.glob('*.wav'))
    if not wav_files:
        raise FileNotFoundError(f'No .wav files found in {folder}')

    choice = questionary.select(
        prompt_text,
        choices=[path.name for path in wav_files],
    ).ask()
    if choice is None:
        raise SystemExit('No file selected')
    return str(folder / choice)

operation = questionary.select('Select operation:', choices=['Generate signal', 'Record signal', 'Correlate signals', 'Channel estimation', 'Normal operation', 'Exit']).ask()

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

if operation == 'Generate signal':
    #Generate signal and save as wav file following param specification
    from transmit import main as transmit_main

    transmit_main(params)

elif operation == 'Record signal':
    #File to run a recording function that records audio and saves as wav file 
    from recieve import main as receive_main

    receive_main(params)

#All other are for post processing. Apart from normal operation which is end game program that runs all
elif operation == 'Correlate signals':
    
    #Pick the received and transmitted chirp wav files, 
    #Analyse using plots to determine the synchronization index

elif operation == 'Channel estimation':
    #Pick the received and transmitted chirp wav files, 
    #Analyse using plots to determine the synchronization index
    #Call Channel estimation file

elif operation == 'Normal operation':
    #Record signal, synchronise, estimate channel and recover signal
    

elif operation == 'Exit':
    print('Exiting...')
    raise SystemExit()