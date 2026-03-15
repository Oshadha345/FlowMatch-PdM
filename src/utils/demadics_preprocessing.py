from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEMADICS_WINDOW_SIZE = 2048
DEMADICS_FAULT_CENTER_STRIDE = 8
DEMADICS_MAX_WINDOWS_PER_EVENT = 16
DEMADICS_NORMAL_WINDOW_STRIDE = 2048
DEMADICS_RANDOM_STATE = 42

DEMADICS_CLASS_TO_INDEX = {
    "normal": 0,
    "f16": 1,
    "f17": 2,
    "f18": 3,
    "f19": 4,
}

DEMADICS_INDEX_TO_CLASS = {index: name for name, index in DEMADICS_CLASS_TO_INDEX.items()}

DEMADICS_COLUMNS = [
    "timestamp",
    "P51_05",
    "P51_06",
    "T51_01",
    "F51_01",
    "LC51_03CV",
    "LC51_03X",
    "LC51_03PV",
    "TC51_05",
    "T51_08",
    "D51_01",
    "D51_02",
    "F51_02",
    "PC51_01",
    "T51_06",
    "P51_03",
    "T51_07",
    "P57_03",
    "P57_04",
    "T57_03",
    "FC57_03PV",
    "FC57_03CV",
    "FC57_03X",
    "P74_00",
    "P74_01",
    "T74_00",
    "F74_00",
    "LC74_20CV",
    "LC74_20X",
    "LC74_20PV",
    "F74_30",
    "P74_30",
    "T74_30",
]

DEMADICS_FAULT_EVENTS = [
    {"item": 1, "date": "30102001", "actuator": 1, "fault_tag": "f18", "start": 58800, "end": 59800},
    {"item": 2, "date": "09112001", "actuator": 1, "fault_tag": "f16", "start": 57275, "end": 57550},
    {"item": 3, "date": "09112001", "actuator": 1, "fault_tag": "f18", "start": 58830, "end": 58930},
    {"item": 4, "date": "09112001", "actuator": 1, "fault_tag": "f18", "start": 58520, "end": 58625},
    {"item": 5, "date": "17112001", "actuator": 1, "fault_tag": "f18", "start": 54600, "end": 54700},
    {"item": 6, "date": "17112001", "actuator": 1, "fault_tag": "f16", "start": 56670, "end": 56770},
    {"item": 7, "date": "20112001", "actuator": 1, "fault_tag": "f17", "start": 37780, "end": 38400},
    {"item": 8, "date": "17112001", "actuator": 2, "fault_tag": "f17", "start": 53780, "end": 53794},
    {"item": 9, "date": "17112001", "actuator": 2, "fault_tag": "f17", "start": 54193, "end": 54215},
    {"item": 10, "date": "17112001", "actuator": 2, "fault_tag": "f19", "start": 55482, "end": 55517},
    {"item": 11, "date": "17112001", "actuator": 2, "fault_tag": "f19", "start": 55977, "end": 56015},
    {"item": 12, "date": "17112001", "actuator": 2, "fault_tag": "f19", "start": 57030, "end": 57072},
    {"item": 13, "date": "20112001", "actuator": 2, "fault_tag": "f17", "start": 44400, "end": 86399},
    {"item": 14, "date": "30102001", "actuator": 3, "fault_tag": "f18", "start": 57340, "end": 57890},
    {"item": 15, "date": "09112001", "actuator": 3, "fault_tag": "f16", "start": 60650, "end": 60700},
    {"item": 16, "date": "09112001", "actuator": 3, "fault_tag": "f16", "start": 60870, "end": 60960},
    {"item": 17, "date": "17112001", "actuator": 3, "fault_tag": "f16", "start": 57475, "end": 57530},
    {"item": 18, "date": "17112001", "actuator": 3, "fault_tag": "f16", "start": 57675, "end": 57800},
    {"item": 19, "date": "17112001", "actuator": 3, "fault_tag": "f19", "start": 58150, "end": 58325},
]


def demadics_paths(repo_root: Path) -> Dict[str, Path]:
    workspace_root = repo_root.parent
    return {
        "archive_dir": workspace_root / "datasets" / "Damadics",
        "raw_dir": workspace_root / "datasets" / "damadics_raw" / "Lublin_all_data",
        "processed_dir": workspace_root / "datasets" / "processed" / "demadics",
    }


def extract_demadics_archives(archive_dir: Path, raw_parent_dir: Path) -> Path:
    raw_parent_dir.mkdir(parents=True, exist_ok=True)
    archive_paths = sorted(archive_dir.glob("Lublin_all_data_part*.zip"))
    if not archive_paths:
        raise FileNotFoundError(f"No DEMADICS part archives found in {archive_dir}")

    expected_dir = raw_parent_dir / "Lublin_all_data"
    if len(list(expected_dir.glob("*.txt"))) >= 25:
        return expected_dir

    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(raw_parent_dir)

    day_files = sorted(expected_dir.glob("*.txt"))
    if len(day_files) != 25:
        raise RuntimeError(f"Expected 25 DEMADICS daily files after extraction, found {len(day_files)}.")
    return expected_dir


def load_demadics_day(day_path: Path) -> np.ndarray:
    frame = pd.read_csv(
        day_path,
        sep="\t",
        header=None,
        names=DEMADICS_COLUMNS,
        dtype=np.float32,
        na_values=["NaN"],
    )
    if frame.shape != (86400, 33):
        raise RuntimeError(f"{day_path.name} has unexpected shape {frame.shape}, expected (86400, 33).")
    if frame.isna().any().any():
        frame = frame.interpolate(limit_direction="both").ffill().bfill()
    timestamps = frame["timestamp"].to_numpy(dtype=np.int32, copy=False)
    if timestamps[0] != 0 or timestamps[-1] != 86399:
        raise RuntimeError(f"{day_path.name} timestamps must span [0, 86399], got [{timestamps[0]}, {timestamps[-1]}].")
    if not np.array_equal(timestamps, np.arange(86400, dtype=np.int32)):
        raise RuntimeError(f"{day_path.name} timestamps are not strictly monotonic by one-second increments.")
    return frame.to_numpy(dtype=np.float32, copy=True)


def _valid_window_bounds(center: int, total_length: int, window_size: int) -> Tuple[int, int]:
    half_window = window_size // 2
    start = center - half_window
    end = start + window_size
    if start < 0:
        end -= start
        start = 0
    if end > total_length:
        start -= end - total_length
        end = total_length
    return start, end


def _sample_centers(start: int, end: int, stride: int, max_centers: int, min_center: int, max_center: int) -> List[int]:
    start = max(start, min_center)
    end = min(end, max_center)
    if start > end:
        return []

    centers = list(range(start, end + 1, stride))
    if not centers:
        centers = [int((start + end) // 2)]
    if len(centers) > max_centers:
        sampled_idx = np.linspace(0, len(centers) - 1, num=max_centers, dtype=int)
        centers = [centers[i] for i in sampled_idx]
    return sorted(set(int(center) for center in centers))


def build_demadics_dataset(
    raw_dir: Path,
    window_size: int = DEMADICS_WINDOW_SIZE,
    fault_center_stride: int = DEMADICS_FAULT_CENTER_STRIDE,
    max_windows_per_event: int = DEMADICS_MAX_WINDOWS_PER_EVENT,
    normal_window_stride: int = DEMADICS_NORMAL_WINDOW_STRIDE,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    day_cache: Dict[str, np.ndarray] = {}

    def day_array(date_code: str) -> np.ndarray:
        if date_code not in day_cache:
            day_cache[date_code] = load_demadics_day(raw_dir / f"{date_code}.txt")
        return day_cache[date_code]

    fault_windows: List[np.ndarray] = []
    fault_labels: List[int] = []
    fault_metadata: List[Dict[str, object]] = []

    min_center = window_size // 2
    max_center = 86400 - (window_size // 2) - 1

    for event in DEMADICS_FAULT_EVENTS:
        centers = _sample_centers(
            event["start"],
            event["end"],
            stride=fault_center_stride,
            max_centers=max_windows_per_event,
            min_center=min_center,
            max_center=max_center,
        )
        day_data = day_array(event["date"])
        for center in centers:
            start, end = _valid_window_bounds(center, total_length=day_data.shape[0], window_size=window_size)
            window = day_data[start:end, 1:]
            if window.shape != (window_size, len(DEMADICS_COLUMNS) - 1):
                raise RuntimeError(f"Fault window extraction failed for event {event['item']} at center {center}.")
            fault_windows.append(window)
            fault_labels.append(DEMADICS_CLASS_TO_INDEX[event["fault_tag"]])
            fault_metadata.append(
                {
                    "item": event["item"],
                    "date": event["date"],
                    "fault_tag": event["fault_tag"],
                    "actuator": event["actuator"],
                    "center": center,
                    "window_start": start,
                    "window_end": end - 1,
                }
            )

    fault_dates = {event["date"] for event in DEMADICS_FAULT_EVENTS}
    healthy_dates = sorted(path.stem for path in raw_dir.glob("*.txt") if path.stem not in fault_dates)

    normal_candidates: List[Tuple[str, int]] = []
    for date_code in healthy_dates:
        total_length = day_array(date_code).shape[0]
        for start in range(0, total_length - window_size + 1, normal_window_stride):
            normal_candidates.append((date_code, start))

    target_normal_count = len(fault_windows)
    if len(normal_candidates) < target_normal_count:
        raise RuntimeError(
            f"Not enough healthy DEMADICS candidate windows: need {target_normal_count}, got {len(normal_candidates)}."
        )

    normal_indices = np.linspace(0, len(normal_candidates) - 1, num=target_normal_count, dtype=int)
    normal_windows: List[np.ndarray] = []
    normal_labels: List[int] = []
    normal_metadata: List[Dict[str, object]] = []
    for idx in normal_indices:
        date_code, start = normal_candidates[int(idx)]
        day_data = day_array(date_code)
        end = start + window_size
        window = day_data[start:end, 1:]
        if window.shape != (window_size, len(DEMADICS_COLUMNS) - 1):
            raise RuntimeError(f"Healthy window extraction failed for {date_code} at start {start}.")
        normal_windows.append(window)
        normal_labels.append(DEMADICS_CLASS_TO_INDEX["normal"])
        normal_metadata.append(
            {
                "item": None,
                "date": date_code,
                "fault_tag": "normal",
                "actuator": None,
                "center": start + (window_size // 2),
                "window_start": start,
                "window_end": end - 1,
            }
        )

    X = np.asarray(normal_windows + fault_windows, dtype=np.float32)
    y = np.asarray(normal_labels + fault_labels, dtype=np.int64)
    metadata = {
        "window_size": window_size,
        "num_features": X.shape[-1],
        "normal_window_stride": normal_window_stride,
        "fault_center_stride": fault_center_stride,
        "max_windows_per_event": max_windows_per_event,
        "class_to_index": DEMADICS_CLASS_TO_INDEX,
        "index_to_class": DEMADICS_INDEX_TO_CLASS,
        "feature_names": DEMADICS_COLUMNS[1:],
        "healthy_dates": healthy_dates,
        "fault_dates": sorted(fault_dates),
        "fault_events": DEMADICS_FAULT_EVENTS,
        "normal_windows": normal_metadata,
        "fault_windows": fault_metadata,
    }
    return X, y, metadata


def stratified_split_and_scale(X: np.ndarray, y: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[str, object]]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=DEMADICS_RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=DEMADICS_RANDOM_STATE,
    )

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    def transform(split_X: np.ndarray) -> np.ndarray:
        reshaped = split_X.reshape(-1, split_X.shape[-1])
        scaled = scaler.transform(reshaped)
        return scaled.reshape(split_X.shape).astype(np.float32)

    X_train = transform(X_train)
    X_val = transform(X_val)
    X_test = transform(X_test)

    split_metadata = {
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "train_class_counts": {DEMADICS_INDEX_TO_CLASS[int(k)]: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "val_class_counts": {DEMADICS_INDEX_TO_CLASS[int(k)]: int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
        "test_class_counts": {DEMADICS_INDEX_TO_CLASS[int(k)]: int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), split_metadata


def prepare_demadics_processed(
    repo_root: Path,
    force_rebuild: bool = False,
    window_size: int = DEMADICS_WINDOW_SIZE,
) -> Dict[str, object]:
    paths = demadics_paths(repo_root)
    processed_dir = paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        processed_dir / "X_train.npy",
        processed_dir / "y_train.npy",
        processed_dir / "X_val.npy",
        processed_dir / "y_val.npy",
        processed_dir / "X_test.npy",
        processed_dir / "y_test.npy",
        processed_dir / "metadata.json",
    ]
    if not force_rebuild and all(path.exists() for path in required_files):
        return json.loads((processed_dir / "metadata.json").read_text())

    raw_dir = extract_demadics_archives(paths["archive_dir"], paths["raw_dir"].parent)
    X, y, metadata = build_demadics_dataset(raw_dir=raw_dir, window_size=window_size)
    train_split, val_split, test_split, split_metadata = stratified_split_and_scale(X, y)

    np.save(processed_dir / "X_train.npy", train_split[0])
    np.save(processed_dir / "y_train.npy", train_split[1])
    np.save(processed_dir / "X_val.npy", val_split[0])
    np.save(processed_dir / "y_val.npy", val_split[1])
    np.save(processed_dir / "X_test.npy", test_split[0])
    np.save(processed_dir / "y_test.npy", test_split[1])

    full_metadata = {
        **metadata,
        **split_metadata,
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
    }
    (processed_dir / "metadata.json").write_text(json.dumps(full_metadata, indent=2))
    return full_metadata
