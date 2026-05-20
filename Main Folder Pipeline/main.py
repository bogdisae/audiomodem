import questionary


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

operation = questionary.select('Select operation:', choices=['Generate signal', 'Record signal', 'Compare signals', 'Exit']).ask()

params = {
        'key_type': 'chirp',
        'repeat_key_count': 1,
        'block_length': 1024,
        'cyclic_prefix_length': 32,
        'length': 50000,
        'fs': 44100, #Generating signal
        'fs_record': 44100, #Recording signal
        'silence_duration': 0.0

    }

if operation == 'Generate signal':
    #Generate signal and save as wav file following param specification
    
elif operation == 'Record signal':
    #File to run a recording function that records audio and saves as wav file 

elif operation == 'Correlate signals':
    
    #Pick the received and transmitted chirp wav files, 
    #Analyse using plots to determine the synchronization index

elif operation == ''
    

elif operation == 'Exit':
    print('Exiting...')
    raise SystemExit()