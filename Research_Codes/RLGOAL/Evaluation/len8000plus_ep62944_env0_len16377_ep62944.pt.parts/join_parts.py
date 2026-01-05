#!/usr/bin/env python3
"""
join_parts.py

Join .part### files produced by split_parts.py and verify integrity against the
manifest's total_bytes and sha256.

Example:
  python3 join_parts.py \
      --parts_dir "$HOME/Research_Codes/RLGOAL/Evaluation/len8000plus_ep62944_env0_len16377_ep62944.pt.parts" \
      --out_file  "$HOME/Research_Codes/RLGOAL/Evaluation/len8000plus_ep62944_env0_len16377_ep62944.pt"

"""

import argparse
import hashlib
from pathlib import Path

BUF = 8 * 1024 * 1024  # 8 MiB streaming buffer

def parse_manifest(manifest_path: Path) -> dict:
    d = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def join_file(parts_dir: Path, out_file: Path | None) -> None:
    parts_dir = parts_dir.resolve()

    manifests = sorted(parts_dir.glob("*.manifest.txt"))
    manifest = manifests[0] if manifests else None
    if not manifest:
        raise FileNotFoundError(f"No *.manifest.txt found in {parts_dir}")

    meta = parse_manifest(manifest)
    base = meta["filename"]
    num_parts = int(meta["num_parts"])
    expected_total = int(meta["total_bytes"])
    expected_sha256 = meta.get("sha256")

    part_paths = [parts_dir / f"{base}.part{i:03d}" for i in range(num_parts)]
    missing = [p for p in part_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing part(s): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    if out_file is None:
        out_file = parts_dir / base
    out_file = Path(out_file).resolve()

    print(f"Joining {num_parts} parts -> {out_file}")
    with out_file.open("wb") as out:
        for p in part_paths:
            with p.open("rb") as pf:
                while True:
                    buf = pf.read(BUF)
                    if not buf:
                        break
                    out.write(buf)

    actual = out_file.stat().st_size
    print(f"Output size: {actual} bytes")
    if actual != expected_total:
        raise RuntimeError(f"Size mismatch: expected {expected_total}, got {actual}")
    print("Size check: OK")

    if expected_sha256:
        actual_sha256 = sha256_file(out_file)
        print(f"SHA256 expected: {expected_sha256}")
        print(f"SHA256 actual:   {actual_sha256}")
        if actual_sha256 != expected_sha256:
            raise RuntimeError("SHA256 mismatch (file corrupted or wrong parts).")
        print("SHA256 check: OK")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts_dir", required=True, help="Directory containing .part### files + manifest")
    ap.add_argument("--out_file", default=None, help="Output file path (optional)")
    args = ap.parse_args()

    join_file(Path(args.parts_dir), Path(args.out_file) if args.out_file else None)

if __name__ == "__main__":
    main()
