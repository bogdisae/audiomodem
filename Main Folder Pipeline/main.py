import questionary
from pathlib import Path

base_dir = Path(__file__).parent

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


operation = questionary.select('Select operation:', choices=['Generate signal', 'Receive signal', 'Correlate signals', 'Channel estimation', 'Normal operation', 'Exit']).ask()
params = {
        # MAYBE ADD CHIRP PARAMATERS E.G CHIRP LENGTH, START AND END FREQUENCIES - SAM
        'key_type': 'chirp', #up_down_chirp
        'length_of_key': 12000, # length of key 
        'f0': 100, #Start frequency of chirp
        'f1': 4000, #End frequency of chirp
        'block_length': 1024,
        'cyclic_prefix_length': 128,
        'read_prefix_early_samples': 30, # Deliberately read some samples before the detected sync index 
        'fs': 48000, # GLOBAL sample rate
        'modulation_scheme': 'QPSK'
}

if operation == 'Generate signal':
    #Generate signal and save as wav file following param specification
    from transmit import main as transmit_main
    transmit_main(params)

elif operation == 'Receive signal':
    #File to run a receive function that allows you to either record or pick an existing file
    from receive import main as receive_main

    #Possible recording rates
    #Aaron's laptop: 48 000Hz
    #Sam's laptop: 44100 or 48000
    #Bogdan's laptop: 

    receive_main(params)

#All other are for post processing. Apart from normal operation which is end game program that runs all
elif operation == 'Correlate signals' or operation == 'Channel estimation':
    from receive_functions import key_matched_filter, normalise_signal, save_wav_file, generate_key
    from scipy.io import wavfile
    from channel_estimation import isolate_key_signal, estimate_channel_response
    #For assessing synchronisation
    Audio_path = base_dir / 'Audio Files'
    rx_file_path = pick_wav_file('Select received wav file:', Audio_path)
    
    #Checks if the key/preamble wav file already exists, if not runs the function to create it
    key_wav_exist = questionary.confirm('Do you already have the correct digital key wav file?').ask()
    key_length_seconds = params['length_of_key'] / params['fs']

    if key_wav_exist:
        tx_file_path = pick_wav_file('Select transmitted key only wav file:', Audio_path)
    else:
        #Create digital key wav file using the same parameters as the transmitted signal
        
        key = generate_key(params['fs'], key_length_seconds, params['f0'], params['f1'], params['key_type'])
        file_name = 'generated_' + params['key_type'] + '_' + str(params['f0']) + '_' + str(params['f1']) + '_' + str(params['fs']) + '_' + str(round(key_length_seconds,2)) + '.wav'
        
        save_wav_file(key, params['fs'], Audio_path / file_name)
        tx_file_path = Audio_path / file_name

    #Find sync index    
    
    try:   
        sync_index = key_matched_filter(normalise_signal(wavfile.read(rx_file_path)[1]), params['fs_record'], key_length_seconds, params['f0'], params['f1'], key_type=params['key_type'])
    except ValueError as e:
        print("Error during matched filtering:", e)
        raise
    
    print(f"{params['key_type']} starts at sample:", sync_index)

    if operation == 'Channel estimation':
        #Pick the received and transmitted chirp wav files, 
        #Analyse using plots to determine the synchronization index
        #Call Channel estimation file
        isolated_key_path = isolate_key_signal(rx_file_path, sync_index, params)

        #undecided what type of response is returned here
        #Cutting off silence at the end of the digital key/preamble
        channel_response = estimate_channel_response(wavfile.read(isolated_key_path)[1], normalise_signal(wavfile.read(tx_file_path)[1][:params['length_of_key']]), params)

    pass
    #Pick the received and transmitted chirp wav files, 
    #Analyse using plots to determine the synchronization index

elif operation == 'Normal operation':
    pass
    #Record signal, synchronise, estimate channel and recover signal
    

elif operation == 'Exit':
    print('Exiting...')
    raise SystemExit()
