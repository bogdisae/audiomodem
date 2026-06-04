from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import questionary


DEFAULT_RESULTS = Path(__file__).resolve().parent / "auto_test"


def pick_results_csv() -> Path:
    default_csv = DEFAULT_RESULTS / "Chirp_count_sweep" / "results" / "test_runs.csv"
    path_text = questionary.text(
        "Path to test_runs.csv",
        default=str(default_csv),
    ).ask()
    if path_text is None:
        raise SystemExit("No results file selected")
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return path


def parse_blocks_ber(value) -> list[float]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
        return []
    if isinstance(value, (np.ndarray, tuple)):
        return [float(x) for x in value]
    return []


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "blocks_ber" in df.columns:
        df["blocks_ber_list"] = df["blocks_ber"].apply(parse_blocks_ber)
    else:
        df["blocks_ber_list"] = [[] for _ in range(len(df))]

    if "sweep_index" in df.columns:
        df = df.sort_values("sweep_index").reset_index(drop=True)
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def plot_overall_ber(df: pd.DataFrame) -> None:
    labels = df["tx_name"].astype(str).tolist() if "tx_name" in df.columns else [str(i) for i in range(len(df))]
    values = df["ber_overall"].astype(float).to_numpy()

    plt.figure(figsize=(12, 5))
    x = np.arange(len(values))
    plt.plot(x, values, marker="o", linewidth=2)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("BER")
    plt.xlabel("File / sweep order")
    plt.title("Overall BER by file")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_block_ber_ridgeline(df: pd.DataFrame) -> None:
    if "blocks_ber_list" not in df.columns:
        raise ValueError("No per-block BER data found in the CSV")

    max_len = max((len(v) for v in df["blocks_ber_list"]), default=0)
    if max_len == 0:
        raise ValueError("All rows have empty blocks_ber data")

    plt.figure(figsize=(14, max(6, 0.45 * len(df) + 2)))
    x_positions = np.arange(len(df))
    cmap = plt.get_cmap("viridis")

    for idx, (_, row) in enumerate(df.iterrows()):
        blocks = np.asarray(row["blocks_ber_list"], dtype=float)
        if blocks.size == 0:
            continue

        y = np.arange(blocks.size)
        x_center = np.full_like(y, x_positions[idx], dtype=float)

        # Symmetric profile around each file index: BER becomes the half-width.
        x_left = x_center - blocks
        x_right = x_center + blocks

        plt.fill_betweenx(
            y,
            x_left,
            x_right,
            color=cmap(idx / max(1, len(df) - 1)),
            alpha=0.65,
            linewidth=0,
        )
        plt.plot(x_left, y, color="black", linewidth=0.7)
        plt.plot(x_right, y, color="black", linewidth=0.7)

    labels = df["tx_name"].astype(str).tolist() if "tx_name" in df.columns else [str(i) for i in range(len(df))]
    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.xlabel("File / sweep order")
    plt.ylabel("Block index")
    plt.title("Per-block BER symmetric profile plot")
    plt.grid(True, axis="y", alpha=0.25)

    # Reference annotation for one point on the graph.
    first_valid = None
    for idx, (_, row) in enumerate(df.iterrows()):
        blocks = np.asarray(row["blocks_ber_list"], dtype=float)
        if blocks.size:
            first_valid = (idx, 0, float(blocks[0]))
            break
    if first_valid is not None:
        file_idx, block_idx, ber_value = first_valid
        plt.scatter([x_positions[file_idx]], [block_idx], color="red", s=30, zorder=5)
        plt.annotate(
            f"ref: file={file_idx}, block={block_idx}, BER={ber_value:.3g}",
            xy=(x_positions[file_idx], block_idx),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    plt.tight_layout()
    plt.show()


def main() -> None:
    csv_path = pick_results_csv()
    df = load_results(csv_path)

    if df.empty:
        raise SystemExit("No results found in the CSV")

    print(f"Loaded {len(df)} test runs from {csv_path}")
    print(df[[col for col in ["tx_name", "rx_name", "ber_overall"] if col in df.columns]].head())

    plot_overall_ber(df)
    plot_block_ber_ridgeline(df)


if __name__ == "__main__":
    main()
