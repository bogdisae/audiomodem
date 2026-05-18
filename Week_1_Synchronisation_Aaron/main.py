import questionary
from file_functions import save_wav_file

import importlib.util
from pathlib import Path


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

operation = questionary.select('Select operation:', choices=['Generate key', 'Record signal', 'Compare signals', 'Exit']).ask()

if operation == 'Generate key':
    from Generator_key_only import main as generate_key_main
    generate_key_main()
elif operation == 'Record signal':
    from Recieving_signal import record_audio
    duration = 20 #Length of recording
    fs = 44100 #Sampling frequency
    channel = 1
    audio = record_audio(duration, fs=fs, channels=channel)
    save_wav_file(audio, fs, "recording.wav")
elif operation == 'Compare signals':
    base_dir = Path(__file__).parent
    file_1 = pick_wav_file('Select received chirp wav file:', base_dir)
    file_2 = pick_wav_file('Select transmitted chirp wav file:', base_dir)

    #Direct import from this file crashes
    chirp_module = load_module_from_path(
        'recieve_chirp_copy',
        base_dir / 'Recieve_chirp_copy.py',
    )
    receive_chirp_main = getattr(chirp_module, 'main', None)
    if receive_chirp_main is None:
        raise ImportError('Recieve_chirp_copy.py does not define main(wav_file_1, wav_file_2)')

    receive_chirp_main(file_1, file_2)
elif operation == 'Exit':
    print('Exiting...')
    raise SystemExit()