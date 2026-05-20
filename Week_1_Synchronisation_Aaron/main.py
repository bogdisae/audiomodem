import questionary
from file_functions import save_wav_file
import channel_estimation

import importlib.util
from pathlib import Path
base_dir = Path(__file__).parent
from Generator_key_only import generate_key
from channel_estimation import isolate_key_signal, estimate_channel_response

def load_module_from_path(module_name: str, file_path: str | Path):
    path = Path(file_path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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
    from Generator_key_only import main as generate_key_main
    
    sending_file = pick_wav_file('Select received chirp wav file:', base_dir)
    generate_key_main(params, test_signal_wav=sending_file)
    
elif operation == 'Record signal':
    from Recieving_signal import record_audio
    duration = 30 #Length of recording
    channel = 1
    audio = record_audio(duration, fs=params['fs_record'], channels=channel)
    save_wav_file(audio, params['fs_record'], "recording.wav")

elif operation == 'Compare signals':
    
    r_file = pick_wav_file('Select received chirp wav file:', base_dir)
    t_file = pick_wav_file('Select transmitted chirp wav file:', base_dir)

    #Direct import from this file crashes
    chirp_module = load_module_from_path(
        'recieve_chirp_copy',
        base_dir / 'Recieve_chirp_copy.py',
    )
    receive_chirp_main = getattr(chirp_module, 'main', None)
    if receive_chirp_main is None:
        raise ImportError('Recieve_chirp_copy.py does not define main(wav_file_1, wav_file_2)')

    sync_idx = receive_chirp_main(r_file, t_file)

    #Call here to recover channel params and orignal signal
    isolated_key = isolate_key_signal(r_file, sync_idx, params)
    key = generate_key(params['length'], params['fs'], params['key_type'], params['repeat_key_count'], params['silence_duration'])

    channel_f_response = estimate_channel_response(isolated_key, key, params)

elif operation == 'Exit':
    print('Exiting...')
    raise SystemExit()