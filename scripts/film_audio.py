"""Original ambient soundtrack for the 30-second film, synthesised with NumPy.

Everything is derived from the same timeline the picture uses (read from the page), so the sound follows the
action: typewriter ticks as the words appear, a hiss that thins out as the snow-static clears, a low hit when
the verdict lands, a heartbeat and footsteps in snow during the walk out. No samples, no music from anywhere.
"""

from __future__ import annotations

import wave

import numpy as np

SR = 44_100


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR)) / SR


def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _ramp(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip((t - a) / (b - a), 0, 1)


def _band(noise: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep only the frequencies between `low` and `high` Hz."""
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1 / SR)
    spectrum[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, len(noise))


def _add(track: np.ndarray, at: float, sound: np.ndarray, gain: float = 1.0) -> None:
    i = int(at * SR)
    if i >= len(track):
        return
    end = min(len(track), i + len(sound))
    track[i:end] += sound[: end - i] * gain


def _reverb(x: np.ndarray, seconds: float = 1.6, wet: float = 0.28, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    impulse = rng.standard_normal(n) * np.exp(-np.arange(n) / (SR * seconds / 4.5))
    impulse = _band(impulse, 120, 5000)
    impulse /= np.abs(impulse).sum() / 4
    size = len(x) + n
    wet_signal = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(impulse, size), size)[: len(x)]
    return (1 - wet) * x + wet * wet_signal


def sigma_at(t: float, info: dict, sigma: list[float]) -> float:
    """The static's strength at time t: the same mapping the picture uses (steps eased by u**1.5)."""
    d0, d1 = info["denoise"]
    if t < info["reveal"][0]:
        return 1.0
    u = float(np.clip((t - d0) / (d1 - d0), 0, 1))
    s = (len(sigma) - 1) * u**1.5
    i0 = int(np.floor(s))
    i1 = min(i0 + 1, len(sigma) - 1)
    return sigma[i0] + (sigma[i1] - sigma[i0]) * (s - i0)


def synth(info: dict, sigma: list[float], seconds: float = 30.0) -> np.ndarray:
    """Stereo float32 audio, shape (samples, 2), peak-normalised."""
    rng = np.random.default_rng(1980)
    t = _t(seconds)
    n = len(t)
    mono = np.zeros(n)

    # The drone: a low fifth with a close, uneasy neighbour, breathing slowly. Swells with the denoising.
    breath = 0.85 + 0.15 * np.sin(2 * np.pi * 0.11 * t)
    drone = (
        np.sin(2 * np.pi * 55.0 * t)
        + 0.6 * np.sin(2 * np.pi * 82.4 * t + 0.4 * np.sin(2 * np.pi * 0.07 * t))
        + 0.45 * np.sin(2 * np.pi * 58.3 * t)          # a beating, uneasy neighbour of the root
        + 0.2 * np.sin(2 * np.pi * 110.0 * t)
    )
    d0, d1 = info["denoise"]
    swell = 0.55 + 0.45 * _smooth(_ramp(t, d0 - 1, d1))
    mono += drone * breath * swell * 0.16 * _smooth(_ramp(t, 0.2, 3.0))

    # Wind over snow.
    wind = _band(rng.standard_normal(n), 120, 900)
    wind /= np.abs(wind).max()
    mono += wind * (0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.09 * t + 1.0))) * 0.10 * _smooth(_ramp(t, 0.0, 2.5))

    # Typewriter: a click for every character, a bell at the end of the last line.
    for k, line in enumerate(info["lines"]):
        for i, ch in enumerate(line):
            if ch == " ":
                continue
            at = info["lineStart"][k] + (i + 1) * info["perChar"]
            click = _band(rng.standard_normal(int(0.014 * SR)), 1800, 7000) * np.exp(-np.arange(int(0.014 * SR)) / (SR * 0.003))
            thud = np.sin(2 * np.pi * 140 * np.arange(int(0.03 * SR)) / SR) * np.exp(-np.arange(int(0.03 * SR)) / (SR * 0.008))
            gain = 0.34 * (0.85 + 0.3 * rng.random())
            _add(mono, at, click / np.abs(click).max(), gain * 0.5)
            _add(mono, at, thud, gain * 0.4)

    # Static: a hiss that follows the snow on screen.
    hiss = _band(rng.standard_normal(n), 1500, 9000)
    hiss /= np.abs(hiss).max()
    strength = np.array([0.96 * sigma_at(x, info, sigma) ** 1.3 for x in t[:: SR // 50]])
    strength = np.interp(t, t[:: SR // 50], strength) * _smooth(_ramp(t, info["reveal"][0], info["reveal"][1] + 0.6))
    strength *= 1 - _smooth(_ramp(t, info["fadeOut"][0], info["fadeOut"][1]))
    mono += hiss * strength * 0.16

    # The verdict: a low hit, and a cluster of cold strings that shiver and fade.
    v = info["verdictAt"]
    hit_t = _t(3.0)
    boom = np.sin(2 * np.pi * (46 - 8 * hit_t) * hit_t) * np.exp(-hit_t / 0.7)
    _add(mono, v, boom, 0.9)
    thump = _band(rng.standard_normal(len(hit_t)), 40, 300) * np.exp(-hit_t / 0.25)
    _add(mono, v, thump / np.abs(thump).max(), 0.5)
    shiver_t = _t(5.0)
    shiver = sum(np.sin(2 * np.pi * f * shiver_t + p) for f, p in
                 [(1760, 0), (1864.7, 1.1), (2093, 2.3), (2217.5, 0.4), (3136, 1.9)])
    shiver *= (0.6 + 0.4 * np.sin(2 * np.pi * 5.3 * shiver_t)) * np.exp(-shiver_t / 1.8) * _smooth(shiver_t / 0.05)
    _add(mono, v + 0.02, shiver / np.abs(shiver).max(), 0.16)

    # The walk out: a heartbeat that quickens, and footsteps in snow.
    w0, w1 = info["walk"]
    beat, bt = w0 - 0.4, 0.0
    while beat < w1 + 0.2:
        bpm = 58 + 22 * float(np.clip((beat - w0) / (w1 - w0), 0, 1))
        for off, gain in ((0.0, 1.0), (0.26 * 60 / bpm * 1.9, 0.7)):
            tt = _t(0.22)
            thump = np.sin(2 * np.pi * (62 - 30 * tt) * tt) * np.exp(-tt / 0.06)
            _add(mono, beat + off, thump, 0.55 * gain * _smooth(_ramp(np.array([beat]), w0 - 0.5, w0 + 0.5))[0])
        beat += 60 / bpm
        bt += 1
    step_t, k = w0, 0
    while step_t < w1:
        crunch_len = int(0.11 * SR)
        crunch = _band(rng.standard_normal(crunch_len), 300, 4200) * np.exp(-np.arange(crunch_len) / (SR * 0.03))
        crunch *= 1 + 0.5 * np.sin(np.arange(crunch_len) * 0.31)
        _add(mono, step_t, crunch / np.abs(crunch).max(), 0.26 * (0.8 + 0.4 * rng.random()))
        step_t += 0.46 + 0.02 * (k % 2)
        k += 1

    # Out into black, then a last low chord for the end card.
    mono *= 1 - _smooth(_ramp(t, info["fadeOut"][0], info["fadeOut"][1] + 0.2)) * 0.97
    e0 = info["endCard"][0]
    chord_t = _t(seconds - e0)
    chord = (np.sin(2 * np.pi * 55 * chord_t) + 0.7 * np.sin(2 * np.pi * 82.4 * chord_t) + 0.5 * np.sin(2 * np.pi * 130.8 * chord_t)
             + 0.25 * np.sin(2 * np.pi * 196.0 * chord_t))
    chord *= _smooth(chord_t / 0.9) * (1 - _smooth((chord_t - (seconds - e0 - 0.5)) / 0.5))
    _add(mono, e0, chord, 0.15)

    mono = _reverb(mono)
    left = mono + 0.03 * np.roll(mono, int(0.012 * SR))
    right = mono + 0.03 * np.roll(mono, int(0.017 * SR))
    stereo = np.stack([left, right], axis=1)
    stereo *= 0.89 / np.abs(stereo).max()                       # peak at about -1 dBFS
    fade = np.minimum(1, np.minimum(t / 0.05, (seconds - t) / 0.4))[:, None]  # no clicks at either end
    return (stereo * fade).astype(np.float32)


def write_wav(path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())
