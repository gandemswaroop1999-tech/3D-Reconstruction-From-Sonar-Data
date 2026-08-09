"""
suop_parsers.py
Parsers for the SUOP (Small Underwater Objects 3D Point Cloud) dataset case files.

Each "case" directory (e.g. SUOP_dataset/chair/chair_range_10m/case_002) contains:
    point_cloud.xyz        - calibrated 3D point cloud: X Y Z Intensity (space-sep, CRLF)
    raw_data.son            - proprietary BlueView raw sonar log (binary, magic b'\\x89SON\\r\\n\\x1a\\n')
    metadata/case_settings.txt   - key=value acquisition settings (# comment style)
    metadata/head_info.txt       - "Key: Value" sonar head hardware/calibration params
    metadata/ping_info.csv       - one row per ping (pan/tilt angle, FOV, timestamps, etc.)
    ping_data/ping_N/
        image.pgm            - grayscale sonar image (netpbm P5, ASCII header, binary body)
        image.ppm            - RGB sonar image (netpbm P6, ASCII header, binary body)
        range_data.txt        - Angle,Range,Intensity CSV for that ping's beam sweep

All text files use CRLF line endings (Windows origin) - callers should open with
newline handling that tolerates \\r\\n (default text mode 'r' with universal newlines is fine).
"""
import numpy as np
import pandas as pd
import re
import os


def parse_case_settings(path):
    """Parse metadata/case_settings.txt (# key=value lines) into a flat dict."""
    out = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip().lstrip('#').strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            out[k] = _coerce(v)
    return out


def parse_head_info(path):
    """Parse metadata/head_info.txt ("Key: Value" lines, first line is a title) into a flat dict."""
    out = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines[1:]:  # skip "Head Info" title line
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        k, v = k.strip(), v.strip()
        # values like "0 ms" or "1350000 Hz" -> strip trailing unit for numeric coercion but keep raw too
        out[k + '_raw'] = v
        num_match = re.match(r'^-?\d+\.?\d*', v)
        out[k] = _coerce(num_match.group(0)) if num_match else v
    return out


def parse_ping_info(path):
    """Parse metadata/ping_info.csv into a DataFrame."""
    return pd.read_csv(path)


def parse_range_data(path):
    """Parse a ping's range_data.txt (Angle,Range,Intensity) into a DataFrame."""
    return pd.read_csv(path)


def parse_xyz(path):
    """Parse point_cloud.xyz (space-separated X Y Z Intensity) into an (N,4) float array."""
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr  # columns: x, y, z, intensity


def read_pnm_header(path):
    """
    Minimal netpbm (P5/P6) header reader. Returns (magic, width, height, maxval, data_offset).
    Handles the dataset's fixed-format 'P5 W H MAXVAL ' / 'P6 W H MAXVAL ' single-line ASCII
    header (no embedded comments), consistent with what was observed via hex inspection.
    """
    with open(path, 'rb') as f:
        raw = f.read()
    # tokenize the ASCII header (magic + 3 ints), respecting '#' comments per the PNM spec
    tokens = []
    i = 0
    while len(tokens) < 4:
        while i < len(raw) and raw[i:i+1].isspace():
            i += 1
        if raw[i:i+1] == b'#':
            while i < len(raw) and raw[i:i+1] != b'\n':
                i += 1
            continue
        start = i
        while i < len(raw) and not raw[i:i+1].isspace():
            i += 1
        tokens.append(raw[start:i])
    # exactly one whitespace byte separates the header from binary data per spec
    data_offset = i + 1
    magic = tokens[0].decode('ascii')
    width, height, maxval = (int(t) for t in tokens[1:4])
    return magic, width, height, maxval, data_offset


def read_pgm(path):
    """Read a P5 (binary grayscale) netpbm image into an (H, W) uint8/uint16 array."""
    magic, w, h, maxval, offset = read_pnm_header(path)
    assert magic == 'P5', f"expected P5, got {magic}"
    dtype = np.uint8 if maxval < 256 else '>u2'
    with open(path, 'rb') as f:
        f.seek(offset)
        data = np.frombuffer(f.read(), dtype=dtype)
    return data[:h * w].reshape(h, w)


def read_ppm(path):
    """Read a P6 (binary RGB) netpbm image into an (H, W, 3) uint8/uint16 array."""
    magic, w, h, maxval, offset = read_pnm_header(path)
    assert magic == 'P6', f"expected P6, got {magic}"
    dtype = np.uint8 if maxval < 256 else '>u2'
    with open(path, 'rb') as f:
        f.seek(offset)
        data = np.frombuffer(f.read(), dtype=dtype)
    return data[:h * w * 3].reshape(h, w, 3)


def parse_son_header(path, n_bytes=64):
    """
    Best-effort read of the raw_data.son binary header for provenance/QC only.
    This is Teledyne/BlueView's proprietary MSS log format (magic b'\\x89SON\\r\\n\\x1a\\n',
    an 8-byte PNG-style signature) - it is NOT decoded into point data here, since
    point_cloud.xyz already ships the calibrated X/Y/Z/intensity product derived from
    this file via BlueView's SDK4.6 (see paper Data Post-Processing section). We only
    verify the magic signature and record file size, for integrity/provenance tracking.
    """
    expected_magic = b'\x89SON\r\n\x1a\n'
    with open(path, 'rb') as f:
        header = f.read(n_bytes)
    return {
        'magic_ok': header[:8] == expected_magic,
        'magic_hex': header[:8].hex(),
        'file_size': os.path.getsize(path),
    }


def _coerce(v):
    """Try int, then float, else return string unchanged."""
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    return v


def load_case(case_dir):
    """
    Load one full case directory into a standardized dict:
        {
          'case_settings': dict,
          'head_info': dict,
          'ping_info': DataFrame,
          'xyz': (N,4) array [x,y,z,intensity],
          'son_header': dict,
          'pings': {ping_idx: {'range_data': DataFrame, 'pgm_shape':.., 'ppm_shape':..}}
        }
    Image arrays are NOT loaded by default (kept out to save memory across 120 cases) -
    use read_pgm/read_ppm directly per-ping if needed.
    """
    case_settings = parse_case_settings(os.path.join(case_dir, 'metadata', 'case_settings.txt'))
    head_info = parse_head_info(os.path.join(case_dir, 'metadata', 'head_info.txt'))
    ping_info = parse_ping_info(os.path.join(case_dir, 'metadata', 'ping_info.csv'))
    xyz = parse_xyz(os.path.join(case_dir, 'point_cloud.xyz'))
    son_header = parse_son_header(os.path.join(case_dir, 'raw_data.son'))

    pings = {}
    ping_root = os.path.join(case_dir, 'ping_data')
    if os.path.isdir(ping_root):
        for ping_name in sorted(os.listdir(ping_root)):
            ping_dir = os.path.join(ping_root, ping_name)
            if not os.path.isdir(ping_dir):
                continue
            idx = int(ping_name.split('_')[-1])
            rd_path = os.path.join(ping_dir, 'range_data.txt')
            entry = {}
            if os.path.exists(rd_path):
                entry['range_data'] = parse_range_data(rd_path)
            pgm_path = os.path.join(ping_dir, 'image.pgm')
            if os.path.exists(pgm_path):
                _, w, h, mv, _ = read_pnm_header(pgm_path)
                entry['pgm_shape'] = (h, w)
            ppm_path = os.path.join(ping_dir, 'image.ppm')
            if os.path.exists(ppm_path):
                _, w, h, mv, _ = read_pnm_header(ppm_path)
                entry['ppm_shape'] = (h, w)
            pings[idx] = entry

    return {
        'case_settings': case_settings,
        'head_info': head_info,
        'ping_info': ping_info,
        'xyz': xyz,
        'son_header': son_header,
        'pings': pings,
    }
