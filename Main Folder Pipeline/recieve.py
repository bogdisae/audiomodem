#import relevant libraries
from pathlib import Path
import questionary
from scipy.io import wavfile
from receive_functions import normalise_signal, chirp_matched_filter

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

def main(params):

    # Choose how to access the audio file

    mode = questionary.select("Do you want to record audio or select an existing file?",
        choices=["Record audio", "Select file"]
    ).ask()
    if mode is None: raise SystemExit("No option selected")
    if mode == "Select file":
        selected_path = pick_wav_file("Select a WAV file:", Path("./Main Folder Pipeline/Audio Files"))
    elif mode == "Record audio":
        print("Recording mode selected")
        # TO DO: CALL A RECORDING FUNCTION, OR SOMETHING 
        selected_path = None

    # Load the audio file
    fs_rx, rxSig = wavfile.read(selected_path)
    rxSig = normalise_signal(rxSig)

    # Unsure what sampling rate to use here
    sync_index = chirp_matched_filter(rxSig, params['fs_record'], 1, 100, 8000)

    print("Chirp starts at sample:", sync_index)
    
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