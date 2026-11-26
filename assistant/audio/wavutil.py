"""WAV <-> raw PCM helpers (mono, 16-bit)."""
from __future__ import annotations

import io
import struct
import wave


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm16(data: bytes) -> tuple[bytes, int]:
    """Tolerant RIFF parser.

    Streamed WAVs (e.g. ``espeak-ng --stdout``) can carry bogus/zero chunk sizes
    because the writer couldn't seek back, which makes ``wave.open`` unreliable.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos, rate, channels, bits = 12, None, 1, 16
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            channels = struct.unpack_from("<H", data, body + 2)[0]
            rate = struct.unpack_from("<I", data, body + 4)[0]
            bits = struct.unpack_from("<H", data, body + 14)[0]
        elif chunk_id == b"data":
            if rate is None:
                raise ValueError("data chunk before fmt chunk")
            if channels != 1 or bits != 16:
                raise ValueError(f"unsupported WAV format: {channels}ch/{bits}bit (need mono 16-bit)")
            end = len(data) if size in (0, 0xFFFFFFFF) or body + size > len(data) else body + size
            pcm = data[body:end]
            return pcm[: len(pcm) - (len(pcm) % 2)], rate
        pos = body + size + (size & 1)
    raise ValueError("no data chunk in WAV")
