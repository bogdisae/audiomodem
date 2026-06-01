from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "Data Files" / "test_runs.csv"


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


@dataclass
class TestRunRecord:
    timestamp: str
    sweep_name: str
    sweep_value: str
    sweep_index: int
    tx_name: str
    rx_name: str
    tx_file: str
    rx_file: str
    mode: str
    sample_rate: int
    block_length: int
    cp_length: int
    active_bins: int
    ber_overall: float
    errors: int
    min_len: int
    ber_variance: float = np.nan
    ber_trend_slope: float = np.nan
    blocks_ber: list[float] = field(default_factory=list)
    cfo_sfo_estimates: list[float] = field(default_factory=list)
    h_mean_abs: float = np.nan
    h_min_abs: float = np.nan
    h_max_abs: float = np.nan
    h_nan_count: int = 0
    h_inf_count: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            data[key] = _jsonify(value)
        return data


def create_test_run_record(
    *,
    sweep_name: str,
    sweep_value: str,
    sweep_index: int,
    tx_name: str,
    rx_name: str,
    tx_file: str,
    rx_file: str,
    mode: str,
    sample_rate: int,
    block_length: int,
    cp_length: int,
    active_bins: int,
    ber_overall: float,
    errors: int,
    min_len: int,
    blocks_ber: Optional[list[float]] = None,
    cfo_sfo_estimates: Optional[list[float]] = None,
    h_values: Optional[np.ndarray] = None,
    ber_variance: Optional[float] = None,
    ber_trend_slope: Optional[float] = None,
    notes: str = "",
) -> TestRunRecord:
    h_array = np.asarray(h_values) if h_values is not None else np.asarray([])
    h_abs = np.abs(h_array) if h_array.size else np.asarray([])

    if h_abs.size:
        h_mean_abs = float(np.nanmean(h_abs))
        h_min_abs = float(np.nanmin(h_abs))
        h_max_abs = float(np.nanmax(h_abs))
        h_nan_count = int(np.sum(np.isnan(h_array)))
        h_inf_count = int(np.sum(np.isinf(h_array)))
    else:
        h_mean_abs = np.nan
        h_min_abs = np.nan
        h_max_abs = np.nan
        h_nan_count = 0
        h_inf_count = 0

    return TestRunRecord(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        sweep_name=sweep_name,
        sweep_value=sweep_value,
        sweep_index=int(sweep_index),
        tx_name=tx_name,
        rx_name=rx_name,
        tx_file=tx_file,
        rx_file=rx_file,
        mode=mode,
        sample_rate=sample_rate,
        block_length=block_length,
        cp_length=cp_length,
        active_bins=active_bins,
        ber_overall=float(ber_overall),
        errors=int(errors),
        min_len=int(min_len),
        ber_variance=float(ber_variance) if ber_variance is not None else np.nan,
        ber_trend_slope=float(ber_trend_slope) if ber_trend_slope is not None else np.nan,
        blocks_ber=list(blocks_ber or []),
        cfo_sfo_estimates=list(cfo_sfo_estimates or []),
        h_mean_abs=h_mean_abs,
        h_min_abs=h_min_abs,
        h_max_abs=h_max_abs,
        h_nan_count=h_nan_count,
        h_inf_count=h_inf_count,
        notes=notes,
    )


def append_test_run(record: TestRunRecord, log_path: Path | str = DEFAULT_LOG_PATH) -> Path:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([record.to_dict()])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(log_path, index=False)
    return log_path
