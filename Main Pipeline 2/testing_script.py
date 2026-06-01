from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import questionary
import sounddevice as sd
from scipy.io.wavfile import write

from constellation import Constellation
from equaliser import GolayPairs
from helper import calculate_ber, csv_bytes_to_binary_sequence, csv_to_data_bytes, normalise_signal, pick_csv_file
from proposed_rx import Rx
from proposed_synchroniser import RepeatedChirpSync
from proposed_tx import Tx
from testing_backend import append_test_run, create_test_run_record


SAMPLE_RATE_DEFAULT = 48_000
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "auto_test"


CONSTELLATION = Constellation(
    1,
    {("0",): 1, ("1",): -1},
    {("0",): lambda s: s.real >= 0, ("1",): lambda s: s.real < 0},
    default_pilot=1 + 0j,
)


@dataclass
class SweepSpec:
    sweep_name: str
    sweep_param: str
    sweep_values: list[int]
    fixed: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def capture_audio(fs: int, channels: int = 1) -> np.ndarray:
    print("Press ENTER to start recording...")
    input()
    print("Recording... press ENTER again to stop.")

    frames: list[np.ndarray] = []
    recording = True

    def callback(indata, frames_count, time, status):
        if recording:
            frames.append(indata.copy())

    stream = sd.InputStream(samplerate=fs, channels=channels, dtype="float32", callback=callback)
    stream.start()
    input()

    recording = False
    stream.stop()
    stream.close()

    if not frames:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(frames, axis=0)
    if channels == 1:
        audio = audio.reshape(-1)
    return audio


def ask_int(prompt: str, default: int) -> int:
    answer = questionary.text(prompt, default=str(default)).ask()
    if answer is None or answer.strip() == "":
        return default
    return int(answer)


def parse_values(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        raise ValueError("No sweep values provided")

    if ":" in raw and "," not in raw:
        parts = [int(part.strip()) for part in raw.split(":") if part.strip()]
        if len(parts) == 2:
            start, stop = parts
            step = 1 if stop >= start else -1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError("Use start:stop or start:stop:step")
        if step == 0:
            raise ValueError("Step cannot be zero")
        stop_inclusive = stop + (1 if step > 0 else -1)
        return list(range(start, stop_inclusive, step))

    values = [int(token.strip()) for token in raw.replace(";", ",").split(",") if token.strip()]
    if not values:
        raise ValueError("No valid sweep values parsed")
    return values


def ask_values() -> list[int]:
    raw = questionary.text("Enter sweep values (comma-separated or start:stop:step)", default="0:40000:5000").ask()
    if raw is None:
        raise SystemExit("No sweep values entered")
    return parse_values(raw)


def ask_sweep_spec() -> SweepSpec:
    sweep_param = questionary.select(
        "What do you want to sweep?",
        choices=["TX chirp gap", "Golay pair gap", "Number of chirps", "Number of Golay pairs"],
    ).ask()
    if sweep_param is None:
        raise SystemExit("No sweep selected")

    default_name = {
        "TX chirp gap": "Chirp_gap_sweep",
        "Golay pair gap": "Golay_gap_sweep",
        "Number of chirps": "Chirp_count_sweep",
        "Number of Golay pairs": "Golay_pair_count_sweep",
    }[sweep_param]
    sweep_name = questionary.text("Sweep folder name", default=default_name).ask()
    if sweep_name is None:
        raise SystemExit("No sweep folder name entered")

    data_csv = pick_csv_file("Select the TX data CSV file", Path("./Main Pipeline 2/Data Files"))
    output_root_text = questionary.text(
        "Output root folder (use your local Google Drive synced folder if needed)",
        default=str(DEFAULT_OUTPUT_ROOT),
    ).ask()
    if output_root_text is None:
        raise SystemExit("No output folder entered")

    fixed = {
        "data_csv": data_csv,
        "output_root": output_root_text,
        "sample_rate": ask_int("Sample rate", SAMPLE_RATE_DEFAULT),
        "block_length": ask_int("OFDM block length", 1024),
        "cp_length": ask_int("Cyclic prefix length", 1024),
        "pilot_spacing": ask_int("Pilot spacing (blocks)", 10),
        "key_pilot_samples_spacing": ask_int("Key/pilot spacing after sync (samples)", 1024),
        "chirp_length": ask_int("Chirp length (samples)", 1024),
        "chirp_repeats": ask_int("Number of chirps", 10),
        "chirp_gap": ask_int("Gap between chirps (samples)", 1024),
        "golay_indiv_length": ask_int("Golay indiv length", 1024),
        "golay_gap": ask_int("Golay pair gap (samples)", 10240),
        "golay_pair_count": ask_int("Number of Golay pairs", 1),
        "f0": ask_int("Chirp start frequency", 20),
        "f1": ask_int("Chirp end frequency", 20000),
    }

    return SweepSpec(sweep_name=sweep_name, sweep_param=sweep_param, sweep_values=ask_values(), fixed=fixed)


def build_system(spec: SweepSpec, sweep_value: int):
    fixed = spec.fixed
    chirp_repeats = fixed["chirp_repeats"]
    chirp_gap = fixed["chirp_gap"]
    golay_gap = fixed["golay_gap"]
    golay_pair_count = fixed["golay_pair_count"]

    if spec.sweep_param == "TX chirp gap":
        chirp_gap = sweep_value
    elif spec.sweep_param == "Golay pair gap":
        golay_gap = sweep_value
    elif spec.sweep_param == "Number of chirps":
        chirp_repeats = sweep_value
    elif spec.sweep_param == "Number of Golay pairs":
        golay_pair_count = sweep_value

    synchroniser = RepeatedChirpSync(
        chirp_repeats,
        fixed["chirp_length"],
        chirp_gap,
        fixed["f0"],
        fixed["f1"],
        fixed["sample_rate"],
    )
    equaliser = GolayPairs(
        fixed["golay_indiv_length"],
        golay_gap,
        numPairs=golay_pair_count,
        fs=fixed["sample_rate"],
    )

    tx = Tx(
        constellation=CONSTELLATION,
        data_bytes=csv_to_data_bytes(fixed["data_csv"]),
        equaliser=equaliser,
        synchroniser=synchroniser,
        pilot_config="Block",
        cp_length=fixed["cp_length"],
        block_length=fixed["block_length"],
        pilot_spacing=fixed["pilot_spacing"],
        key_pilot_samples_spacing=fixed["key_pilot_samples_spacing"],
    )
    return tx, equaliser, synchroniser


def generate_mode() -> Path:
    spec = ask_sweep_spec()
    sweep_root = Path(spec.fixed["output_root"]) / spec.sweep_name
    tx_dir = sweep_root / "tx"
    tx_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    print("Generation order:")
    for index, sweep_value in enumerate(spec.sweep_values):
        tx, _, _ = build_system(spec, sweep_value)
        tx.encode()

        tx_file = tx_dir / f"{sweep_value}.wav"
        tx_audio = np.real(tx.transmitted_signal)
        tx_audio = tx_audio / np.max(np.abs(tx_audio)) if np.max(np.abs(tx_audio)) != 0 else tx_audio
        write(tx_file, spec.fixed["sample_rate"], np.int16(np.clip(tx_audio, -1.0, 1.0) * 32767))

        rows.append(
            {
                "sweep_index": index,
                "sweep_name": spec.sweep_name,
                "sweep_param": spec.sweep_param,
                "sweep_value": sweep_value,
                "tx_file": str(tx_file),
                **spec.fixed,
            }
        )
        print(f"  {index + 1}. {tx_file.name} ({spec.sweep_param} = {sweep_value})")

    manifest_path = sweep_root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    (sweep_root / "sweep_spec.json").write_text(spec.to_json(), encoding="utf-8")
    (sweep_root / "instructions.txt").write_text(
        "Play the TX files in manifest order, one at a time.\n"
        "Then run record mode and record them in the same order.\n",
        encoding="utf-8",
    )

    print(f"Saved manifest: {manifest_path}")
    print(f"Folder ready for Google Drive sync: {sweep_root}")
    return manifest_path


def record_one(row: pd.Series, sweep_root: Path) -> None:
    tx_file = Path(row["tx_file"])
    print(f"Play next: {tx_file.name} ({row['sweep_param']} = {row['sweep_value']})")
    audio = capture_audio(int(row["sample_rate"]))
    audio = normalise_signal(audio)

    rx_dir = sweep_root / "rx"
    results_dir = sweep_root / "results"
    rx_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    rx_file = rx_dir / f"{tx_file.stem}_rx.wav"
    write(rx_file, int(row["sample_rate"]), np.int16(np.clip(audio, -1.0, 1.0) * 32767))

    spec = SweepSpec(
        sweep_name=str(row["sweep_name"]),
        sweep_param=str(row["sweep_param"]),
        sweep_values=[int(row["sweep_value"])],
        fixed={k: row[k] for k in [
            "data_csv",
            "output_root",
            "sample_rate",
            "block_length",
            "cp_length",
            "pilot_spacing",
            "key_pilot_samples_spacing",
            "chirp_length",
            "chirp_repeats",
            "chirp_gap",
            "golay_indiv_length",
            "golay_gap",
            "golay_pair_count",
            "f0",
            "f1",
        ]},
    )
    _, equaliser, synchroniser = build_system(spec, int(row["sweep_value"]))

    receiver = Rx(
        CONSTELLATION,
        audio,
        int(row["cp_length"]),
        int(row["block_length"]),
        equaliser,
        synchroniser,
        "Golay",
        "Block",
        pilot_spacing=int(row["pilot_spacing"]),
        key_pilot_samples_spacing=int(row["key_pilot_samples_spacing"]),
    )
    receiver.decode()

    known_bits = csv_bytes_to_binary_sequence(row["data_csv"])
    ber, errors, min_len = calculate_ber(known_bits, receiver.data_bits[: len(known_bits)])

    bits_per_block = len(receiver.active_bins) * CONSTELLATION.bits_per_symbol
    blocks_ber: list[float] = []
    for block_index in range(len(known_bits) // bits_per_block):
        start = block_index * bits_per_block
        end = (block_index + 1) * bits_per_block
        block_ber, _, _ = calculate_ber(known_bits[start:end], receiver.data_bits[start:end])
        blocks_ber.append(block_ber)

    ber_variance = float(np.var(blocks_ber)) if blocks_ber else np.nan
    ber_trend = np.polyfit(range(len(blocks_ber)), blocks_ber, 1) if len(blocks_ber) > 1 else np.array([np.nan, np.nan])

    record = create_test_run_record(
        sweep_name=str(row["sweep_name"]),
        sweep_value=str(row["sweep_value"]),
        sweep_index=int(row["sweep_index"]),
        tx_name=tx_file.stem,
        rx_name=rx_file.stem,
        tx_file=str(tx_file),
        rx_file=str(rx_file),
        mode="recording",
        sample_rate=int(row["sample_rate"]),
        block_length=int(row["block_length"]),
        cp_length=int(row["cp_length"]),
        active_bins=len(receiver.active_bins),
        ber_overall=ber,
        errors=errors,
        min_len=min_len,
        blocks_ber=blocks_ber,
        cfo_sfo_estimates=list(getattr(receiver, "a_history", [])),
        h_values=getattr(receiver, "H", None),
        ber_variance=ber_variance,
        ber_trend_slope=float(ber_trend[0]) if np.isfinite(ber_trend[0]) else np.nan,
        notes=f"{row['sweep_param']}={row['sweep_value']}",
    )
    append_test_run(record, results_dir / "test_runs.csv")
    print(f"Logged {tx_file.name} -> {rx_file.name}")


def record_mode_from_manifest(manifest_path: Path) -> None:
    df = pd.read_csv(manifest_path).sort_values("sweep_index").reset_index(drop=True)
    sweep_root = manifest_path.parent
    print("Recording order:")
    for _, row in df.iterrows():
        print(f"  {int(row['sweep_index']) + 1}. {Path(row['tx_file']).name} ({row['sweep_param']} = {row['sweep_value']})")
    if not questionary.confirm("Start the recording sequence now?").ask():
        raise SystemExit("Recording cancelled")
    for _, row in df.iterrows():
        record_one(row, sweep_root)


def record_mode() -> None:
    manifest_path_text = questionary.text(
        "Path to manifest.csv for this sweep",
        default=str(DEFAULT_OUTPUT_ROOT / "Chirp_gap_sweep" / "manifest.csv"),
    ).ask()
    if manifest_path_text is None:
        raise SystemExit("No manifest entered")
    manifest_path = Path(manifest_path_text)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    record_mode_from_manifest(manifest_path)


def main() -> None:
    mode = questionary.select(
        "Testing script mode",
        choices=["Generate sweep files", "Record sweep files"],
    ).ask()
    if mode is None:
        raise SystemExit("No mode selected")

    if mode == "Generate sweep files":
        manifest_path = generate_mode()
        if questionary.confirm("Proceed to recording mode now?").ask():
            record_mode_from_manifest(manifest_path)
    else:
        record_mode()


if __name__ == "__main__":
    main()
