"""Audio loading / saving. WAV is handled with scipy; everything else through ffmpeg."""
from __future__ import annotations

import io
import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .config import SR


class AudioDecodeError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _to_float_mono(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    else:
        x = data.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def _decode_wav_bytes(buf: bytes):
    try:
        sr, data = wavfile.read(io.BytesIO(buf))
    except Exception:
        return None
    return resample(_to_float_mono(data), sr, SR)


def _decode_ffmpeg(src, buf) -> np.ndarray:
    if not ffmpeg_available():
        raise AudioDecodeError(
            "This audio format needs ffmpeg, which was not found on PATH. "
            "Install ffmpeg (https://ffmpeg.org) or upload a WAV file.")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", src if src else "pipe:0",
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"]
    try:
        proc = subprocess.run(cmd, input=buf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("ffmpeg timed out while decoding the audio") from exc
    out = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if out.size == 0:
        raise AudioDecodeError("Could not decode this audio (corrupt or unsupported format).")
    return out


def load_audio(path) -> np.ndarray:
    """Load any audio file as mono float32 @16 kHz."""
    path = str(path)
    if path.lower().endswith(".wav"):
        with open(path, "rb") as fh:
            x = _decode_wav_bytes(fh.read())
        if x is not None:
            return x
    return _decode_ffmpeg(path, None)


def load_audio_bytes(buf: bytes) -> np.ndarray:
    """Decode an in-memory upload (wav fast-path, ffmpeg for anything else)."""
    if not buf:
        raise AudioDecodeError("Empty audio payload")
    if buf[:4] == b"RIFF":
        x = _decode_wav_bytes(buf)
        if x is not None:
            return x
    return _decode_ffmpeg(None, buf)


def save_wav(path, x: np.ndarray, sr: int = SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.nan_to_num(x), -1.0, 1.0)
    wavfile.write(str(path), sr, (y * 32767.0).astype(np.int16))


def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    bio = io.BytesIO()
    y = np.clip(np.nan_to_num(x), -1.0, 1.0)
    wavfile.write(bio, sr, (y * 32767.0).astype(np.int16))
    return bio.getvalue()
