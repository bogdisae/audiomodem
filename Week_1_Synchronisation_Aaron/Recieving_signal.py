("""Simple recorder helper.

Usage: run the module or import `record_audio(duration_s, fs, channels)`.
Requires the `sounddevice` package: `pip install sounddevice`.

WRITTEN USING COPILOT - adjusted for testing purposes.
""")

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from Generator_key_only import save_wav_file

try:
	import sounddevice as sd
except Exception:  # pragma: no cover - runtime environment
	sd = None


def record_audio(duration_s: float, fs: int = 44100, channels: int = 1, dtype: str = "float32") -> np.ndarray:
	"""Record audio for `duration_s` seconds and return a NumPy array.

	- `fs`: sample rate in Hz
	- `channels`: 1 for mono, 2 for stereo
	- return value: shape (N,) for mono or (N, channels) for multi-channel
	"""
	start_time = datetime.now()
	print(f'Recording began at time: {start_time}')
	print("Recording audio...")
	if sd is None:
		raise ImportError("sounddevice is required for recording. Install with: pip install sounddevice")

	frames = int(round(duration_s * fs))
	print(f"Recording {duration_s:.2f}s @ {fs}Hz, {channels} channel(s)")
	rec = sd.rec(frames, samplerate=fs, channels=channels, dtype=dtype)
	sd.wait()

	arr = np.asarray(rec)
	# If mono, return 1-D array
	if arr.ndim == 2 and arr.shape[1] == 1:
		return arr[:, 0]
	return arr

# Use `save_wav_file` from `Generator_key_only` instead of a local implementation.


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Record audio to WAV")
	parser.add_argument("-d", "--duration", type=float, default=20.0, help="seconds to record")
	parser.add_argument("-r", "--rate", type=int, default=44100, help="sample rate")
	parser.add_argument("-c", "--channels", type=int, default=1, help="channels (1 mono, 2 stereo)")
	parser.add_argument("-o", "--out", type=str, default="recording.wav", help="output filename")

	args = parser.parse_args()
	audio = record_audio(args.duration, fs=args.rate, channels=args.channels)
	# save_wav_file signature: save_wav_file(data, fs, filename)
	save_wav_file(audio, args.rate, args.out)

