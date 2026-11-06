#!/usr/bin/env python3
"""Download free negative audio (no wake word in it) for false-accepts-per-hour testing and hard-negative mining.

    python scripts/fetch_negatives.py --out negatives            # LibriSpeech dev/test-clean + ESC-50 + spoken digits (~11 h)
    python scripts/fetch_negatives.py --out negatives --musan    # + MUSAN speech/music/noise (11 GB, ~100 h)

Then:  python run.py farh-eval --negatives negatives --models models
       python run.py mine-negatives --negatives negatives --models models --apply

Sources and licences: LibriSpeech (OpenSLR 12, CC BY 4.0), ESC-50 (GitHub, CC BY-NC), Free Spoken Digit Dataset (GitHub,
CC BY-SA 4.0), MUSAN (OpenSLR 17, CC BY 4.0 / mixed).  Common Voice needs a free account (commonvoice.mozilla.org):
download an English .tar and unpack its clips/ folder into negatives/common_voice/ - every .mp3 there is picked up.
NOTE: only the GitHub sources were verified to download in the authoring sandbox; the OpenSLR links are the official ones."""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SRC = {
    "librispeech_dev_clean": ("https://www.openslr.org/resources/12/dev-clean.tar.gz", "tar"),
    "librispeech_test_clean": ("https://www.openslr.org/resources/12/test-clean.tar.gz", "tar"),
    "esc50": ("https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip", "zip"),
    "fsdd": ("https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip", "zip"),
    "musan": ("https://www.openslr.org/resources/17/musan.tar.gz", "tar"),
    # added v1.6 (official links, not verified from the authoring sandbox, which could not reach them):
    "librispeech_dev_other": ("https://www.openslr.org/resources/12/dev-other.tar.gz", "tar"),     # noisier / accented read speech
    "librispeech_test_other": ("https://www.openslr.org/resources/12/test-other.tar.gz", "tar"),
    "speech_commands": ("https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz", "tar"),  # 35 words, CC BY 4.0, ~2.3 GB
    "rirs_noises": ("https://www.openslr.org/resources/28/rirs_noises.zip", "zip"),                 # room impulse responses, Apache 2.0
}


def fetch(name, url, kind, out):
    dst = out / name
    if dst.exists() and any(dst.rglob("*")):
        print(f"  {name}: already there"); return
    dst.mkdir(parents=True, exist_ok=True)
    arc = out / f"{name}.{'zip' if kind == 'zip' else 'tar.gz'}"
    print(f"  {name}: downloading {url}")
    urllib.request.urlretrieve(url, arc)
    print(f"  {name}: unpacking")
    if kind == "zip":
        zipfile.ZipFile(arc).extractall(dst)
    else:
        tarfile.open(arc).extractall(dst)
    arc.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="negatives")
    ap.add_argument("--musan", action="store_true")
    ap.add_argument("--extra", action="store_true", help="also Speech Commands v2 and room impulse responses")
    ap.add_argument("--only", default="", help="comma list of: " + ", ".join(SRC))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    OPT = ("musan", "speech_commands", "rirs_noises")
    names = [n for n in (a.only.split(",") if a.only else SRC) if n and (n not in OPT or (n == "musan" and a.musan) or (n != "musan" and a.extra) or a.only)]
    for n in names:
        try:
            fetch(n, *SRC[n], out)
        except Exception as exc:
            print(f"  {n}: FAILED ({exc})")
    print("done ->", out.resolve())


if __name__ == "__main__":
    main()
