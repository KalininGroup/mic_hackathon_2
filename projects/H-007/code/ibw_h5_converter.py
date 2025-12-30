from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py
import sidpy
import pyNSID

# Prefer igor2 (modern). If you already use "igor.binarywave", this still works with igor2.
from igor2 import binarywave as igor_binarywave


def parse_note(note_bytes: bytes | str | None) -> dict:
    """
    Parse Igor wave note into a dict with key:value pairs per line.
    Handles degree (°) and micro (µ) symbol bytes that often appear in IBW notes.
    """
    if note_bytes is None:
        return {}

    if isinstance(note_bytes, str):
        text = note_bytes
    else:
        # Match your source code's replacements
        text = (
            note_bytes
            .replace(b"\xb0", b"\xc2\xb0")
            .replace(b"\xb5", b"\xc2\xb5")
            .decode(errors="replace")
        )

    note = {}
    for line in text.split("\r"):
        s = line.strip()
        if not s or ":" not in s:
            continue
        key, value = s.split(":", 1)
        note[key.strip()] = value.strip()
    return note


def extract_channel_labels(wave: dict, n_channels: int) -> list[str]:
    """
    Try to extract per-channel labels from Igor 'labels' structure.
    Falls back to Channel_000 style if unavailable.
    """
    labels = []
    raw_labels = wave.get("labels", None)

    # Your example uses loadedfile['labels'][2][1:] (skip first placeholder)
    if isinstance(raw_labels, (list, tuple)) and len(raw_labels) >= 3:
        candidate = raw_labels[2]
        if isinstance(candidate, (list, tuple)) and len(candidate) > 1:
            for item in candidate[1:]:
                if isinstance(item, (bytes, bytearray)):
                    labels.append(item.decode(errors="replace"))
                else:
                    labels.append(str(item))
    # If we didn't get enough labels, fill in
    if len(labels) < n_channels:
        for i in range(len(labels), n_channels):
            labels.append(f"Channel_{i:03d}")

    return labels[:n_channels]


def build_sidpy_datasets_from_ibw(ibw_path: Path, rotate_k: int = 1) -> dict[str, sidpy.Dataset]:
    """
    Reads one .ibw and returns a dict of sidpy.Datasets keyed as Channel_00{i}.
    Uses wData[:,:,i] if multi-channel. Uses np.rot90 like your reference code.
    Adds spatial dimensions using note fields when present.
    """
    wave = igor_binarywave.load(str(ibw_path))["wave"]
    image = np.asarray(wave["wData"])

    # note can be under wave['note'] in your snippet; sometimes under wave_note elsewhere
    note_bytes = wave.get("note", None)
    note = parse_note(note_bytes)

    # Determine channels
    if image.ndim == 3:
        n_channels = image.shape[2]
    else:
        n_channels = 1

    labels = extract_channel_labels(wave, n_channels=n_channels)

    # Try to get scan sizes and pixel counts from note; fall back to array shape
    def _to_float(key, default):
        try:
            return float(note.get(key, default))
        except Exception:
            return float(default)

    # After rot90, shape becomes (ScanPoints, ScanLines) if original was (ScanLines, ScanPoints)
    datasets = {}

    for i in range(n_channels):
        arr = image[:, :, i] if n_channels > 1 else image
        arr = np.rot90(arr, k=rotate_k)

        label = labels[i] if i < len(labels) else f"Channel_{i:03d}"
        ds = sidpy.Dataset.from_array(arr, name=label)
        ds.data_type = "image"

        # axis lengths from the rotated array
        nx = int(arr.shape[0])
        ny = int(arr.shape[1])

        fast_size = _to_float("FastScanSize", 1.0)
        slow_size = _to_float("SlowScanSize", 1.0)

        # If the note includes explicit points/lines, you can use them;
        # but using arr.shape is safer after rotation.
        x = np.linspace(0, fast_size, nx)
        y = np.linspace(0, slow_size, ny)

        ds.set_dimension(
            0,
            sidpy.Dimension(x, name="x", units="m", quantity="Length", dimension_type="spatial"),
        )
        ds.set_dimension(
            1,
            sidpy.Dimension(y, name="y", units="m", quantity="Length", dimension_type="spatial"),
        )

        ds.metadata["channel"] = label
        ds.metadata["source_file"] = ibw_path.name
        ds.metadata["note"] = note

        datasets[f"Channel_{i:03d}"] = ds

    return datasets


def save_dataset_dictionary(h5_file: h5py.File, datasets: dict[str, object]):
    """
    Same logic as your reference snippet:
    - creates Measurement_### group
    - writes each sidpy.Dataset as NSID dataset
    """
    h5_measurement_group = sidpy.hdf.prov_utils.create_indexed_group(h5_file, "Measurement_")

    for key, dataset in datasets.items():
        if key.endswith("/"):
            key = key[:-1]

        if isinstance(dataset, sidpy.Dataset):
            h5_group = h5_measurement_group.create_group(key)
            h5_dataset = pyNSID.hdf_io.write_nsid_dataset(dataset, h5_group)
            dataset.h5_dataset = h5_dataset
            h5_dataset.file.flush()

        elif isinstance(dataset, dict):
            sidpy.hdf.hdf_utils.write_dict_to_h5_group(h5_measurement_group, dataset, key)

        else:
            print("Could not save item", key, "of dataset dictionary")

    return h5_measurement_group


def convert_folder_ibw_to_h5(in_folder: str | Path, out_folder: str | Path | None = None):
    in_folder = Path(in_folder)
    if out_folder is None:
        out_folder = Path(str(in_folder) + "_h5")
    else:
        out_folder = Path(out_folder)

    out_folder.mkdir(parents=True, exist_ok=True)

    ibw_files = sorted(in_folder.glob("*.ibw"))
    print(f"Found {len(ibw_files)} .ibw files in {in_folder.resolve()}")
    print(f"Writing .h5 files to {out_folder.resolve()}")

    ok, fail = 0, 0
    for ibw_path in ibw_files:
        try:
            datasets = build_sidpy_datasets_from_ibw(ibw_path)

            out_h5 = out_folder / f"{ibw_path.stem}.h5"
            with h5py.File(out_h5, mode="w") as h5_file:
                save_dataset_dictionary(h5_file, datasets)

            ok += 1
            print(f"[OK]  {ibw_path.name} -> {out_h5.name}")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {ibw_path.name}: {type(e).__name__}: {e}")

    print(f"\nDone. Success: {ok}, Failed: {fail}")


if __name__ == "__main__":
    # Your exact request:
    convert_folder_ibw_to_h5("YW_image", "YW_image_h5_converted")