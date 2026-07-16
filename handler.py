import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import runpod
import soundfile as sf


MAX_BYTES = 30 * 1024 * 1024
MAX_SECONDS = 8 * 60


def download_audio(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "VocalLens/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response, target.open("wb") as output:
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError("Audio file is larger than 30 MB")
            output.write(chunk)


def to_wav(source: Path, target: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-t", str(MAX_SECONDS), "-ac", "1", "-ar", "22050", str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def midi_name(value: float) -> str:
    return librosa.midi_to_note(float(value), unicode=False)


def estimate_key(chroma: np.ndarray) -> str:
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    profile = np.mean(chroma, axis=1)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    scores = []
    for root in range(12):
        scores.append((np.corrcoef(profile, np.roll(major, root))[0, 1], f"{names[root]} Major"))
        scores.append((np.corrcoef(profile, np.roll(minor, root))[0, 1], f"{names[root]} Minor"))
    return max(scores, key=lambda item: item[0])[1]


def analyze(wav_path: Path) -> dict:
    y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=MAX_SECONDS)
    duration = float(librosa.get_duration(y=y, sr=sr))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    f0, voiced, probability = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr, frame_length=2048
    )
    times = librosa.times_like(f0, sr=sr)
    valid = np.isfinite(f0) & voiced & (probability > 0.72)
    midi = librosa.hz_to_midi(f0[valid]) if np.any(valid) else np.array([48, 69])
    low = float(np.percentile(midi, 3))
    high = float(np.percentile(midi, 97))

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=512)
    quiet = rms < max(np.percentile(rms, 22), 0.008)
    breath_points = []
    start = None
    for idx, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = idx
        if (not is_quiet or idx == len(quiet) - 1) and start is not None:
            end = idx
            if 0.12 <= rms_times[end] - rms_times[start] <= 1.8 and rms_times[start] > 2:
                breath_points.append(round(float((rms_times[start] + rms_times[end]) / 2), 2))
            start = None

    transitions = []
    if np.any(valid):
        voiced_times = times[valid]
        voiced_midi = librosa.hz_to_midi(f0[valid])
        threshold = np.percentile(voiced_midi, 70)
        crossed = np.where(np.diff((voiced_midi > threshold).astype(int)) != 0)[0]
        for idx in crossed:
            t = float(voiced_times[idx])
            if not transitions or t - transitions[-1] > 4:
                transitions.append(round(t, 2))

    contour_step = max(1, len(times) // 160)
    contour = [
        {"time": round(float(times[i]), 2), "midi": None if not np.isfinite(f0[i]) else round(float(librosa.hz_to_midi(f0[i])), 1)}
        for i in range(0, len(times), contour_step)
    ]
    return {
        "duration": round(duration, 2),
        "bpm": round(tempo_value),
        "key": estimate_key(chroma),
        "range": {"low": midi_name(low), "high": midi_name(high)},
        "breath_points": breath_points[:24],
        "register_transition_candidates": transitions[:16],
        "pitch_contour": contour,
        "notice": "Breath and register transitions are practice suggestions, not a medical or professional diagnosis.",
    }


def handler(job: dict) -> dict:
    data = job.get("input") or {}
    audio_url = data.get("audio_url")
    if not audio_url or not audio_url.startswith(("https://", "http://")):
        return {"error": "input.audio_url must be an http(s) URL"}
    workdir = Path(tempfile.mkdtemp(prefix="vocal-lens-"))
    try:
        source = workdir / "source.audio"
        wav = workdir / "audio.wav"
        download_audio(audio_url, source)
        to_wav(source, wav)
        return {"ok": True, "analysis": analyze(wav)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
