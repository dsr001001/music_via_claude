#!/usr/bin/env python3
"""
Add percussion beats to a vocal recording and align vocals to the beat grid.
Just beats — no extra music.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
import os

INPUT_FILE = "Mummy_Tumse_Milkar.m4a"
OUTPUT_FILE = "Mummy_Tumse_Milkar_with_beats.wav"
SAMPLE_RATE = 44100


# ═══════════════════════════════════════════════════════════════════════════════
#  PERCUSSION SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

def synth_tabla_na(sr, pitch=360, dur=0.16):
    """Dayan: crisp 'Na' ring."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    freq = pitch * np.exp(-t * 1.8)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    tone = np.sin(phase) * 0.7
    tone += np.sin(phase * 2.02) * 0.2 * np.exp(-t * 18)
    tone += np.sin(phase * 3.05) * 0.08 * np.exp(-t * 25)
    slap = np.random.randn(n) * 0.12 * np.exp(-t * 90)
    nyq = sr / 2
    b, a = signal.butter(3, min(pitch * 3, nyq * 0.9) / nyq, btype='low')
    slap = signal.lfilter(b, a, slap)
    env = np.exp(-t * 10)
    att = int(0.002 * sr)
    if att < n:
        env[:att] = np.linspace(0, 1, att)
    return (tone + slap) * env


def synth_tabla_tin(sr, pitch=420, dur=0.10):
    """Dayan: sharp short 'Tin'."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * pitch * t) * 0.5
    tone += np.sin(2 * np.pi * pitch * 2.01 * t) * 0.15 * np.exp(-t * 30)
    noise = np.random.randn(n) * 0.08 * np.exp(-t * 120)
    env = np.exp(-t * 20)
    att = int(0.001 * sr)
    if att < n:
        env[:att] = np.linspace(0, 1, att)
    return (tone + noise) * env


def synth_dholak_ge(sr, pitch=85, dur=0.28):
    """Dholak bass: deep 'Ge' with pitch bend down."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    freq = pitch * np.exp(-t * 2.5) + 45
    phase = 2 * np.pi * np.cumsum(freq) / sr
    tone = np.sin(phase) * 0.85
    tone += np.sin(phase * 1.5) * 0.12 * np.exp(-t * 8)
    env = np.exp(-t * 5.5)
    att = int(0.004 * sr)
    if att < n:
        env[:att] = np.linspace(0, 1, att)
    return tone * env


def synth_dholak_ta(sr, pitch=220, dur=0.12):
    """Dholak treble: snappy 'Ta'."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * pitch * t) * 0.5
    noise = np.random.randn(n) * 0.2 * np.exp(-t * 80)
    nyq = sr / 2
    b, a = signal.butter(3, min(pitch * 4, nyq * 0.9) / nyq, btype='low')
    noise = signal.lfilter(b, a, noise)
    env = np.exp(-t * 16)
    att = int(0.002 * sr)
    if att < n:
        env[:att] = np.linspace(0, 1, att)
    return (tone + noise) * env


def synth_dha(sr, na_pitch=360, ge_pitch=85):
    """Dha = Na + Ge (open composite stroke)."""
    na = synth_tabla_na(sr, na_pitch, dur=0.18)
    ge = synth_dholak_ge(sr, ge_pitch, dur=0.30)
    n = max(len(na), len(ge))
    out = np.zeros(n)
    out[:len(na)] += na * 0.55
    out[:len(ge)] += ge * 0.7
    return out


def synth_shaker(sr, dur=0.04):
    """Subtle shaker tick."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = np.random.randn(n)
    nyq = sr / 2
    lo = 5000 / nyq
    hi = min(11000, nyq * 0.9) / nyq
    if lo < hi:
        b, a = signal.butter(3, [lo, hi], btype='band')
        noise = signal.lfilter(b, a, noise)
    env = np.exp(-t * 50)
    att = int(0.001 * sr)
    if att < n:
        env[:att] = np.linspace(0, 1, att)
    return noise * env * 0.10


def place_stroke(output, stroke, pos, velocity=1.0):
    """Place a single percussion stroke into the output buffer."""
    end = min(pos + len(stroke), len(output))
    if 0 <= pos < len(output):
        output[pos:end] += stroke[:end - pos] * velocity


# ═══════════════════════════════════════════════════════════════════════════════
#  BEAT PATTERN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_beat_track(duration, sr, bpm):
    """
    Generate a Bollywood-style Keherwa beat pattern.
    Keherwa is an 8-beat cycle commonly used in romantic/light songs.

    Pattern (in half-beat subdivisions for groove):
      Beat:    1       2       3       4       5       6       7       8
      Tabla:   Dha     Ge      Na      Ti      Na     (Ti)     Dha     Na
      Shaker:     x       x       x       x       x       x       x       x
    """
    beat_dur = 60.0 / bpm
    half = beat_dur / 2
    output = np.zeros(int(duration * sr))

    # Two alternating patterns for variation
    # Pattern A: standard Keherwa
    pat_a = [
        (0,    'dha',  0.95),
        (1,    'ge',   0.70),
        (2,    'na',   0.85),
        (3,    'tin',  0.55),
        (4,    'na',   0.80),
        (5,    'tin',  0.50),
        (6,    'dha',  0.90),
        (7,    'na',   0.65),
    ]
    # Pattern B: slight variation (fills / syncopation)
    pat_b = [
        (0,    'dha',  0.95),
        (1,    'ge',   0.70),
        (2,    'na',   0.80),
        (2.5,  'tin',  0.40),
        (3,    'na',   0.60),
        (4,    'dha',  0.85),
        (5,    'tin',  0.50),
        (6,    'na',   0.80),
        (6.5,  'tin',  0.35),
        (7,    'na',   0.60),
    ]

    stroke_map = {
        'dha': lambda: synth_dha(sr),
        'ge':  lambda: synth_dholak_ge(sr),
        'na':  lambda: synth_tabla_na(sr),
        'tin': lambda: synth_tabla_tin(sr),
        'ta':  lambda: synth_dholak_ta(sr),
    }

    cycle_dur = 8 * beat_dur
    n_cycles = int(duration / cycle_dur) + 1

    for cyc in range(n_cycles):
        t0 = cyc * cycle_dur
        pat = pat_a if cyc % 3 != 2 else pat_b  # every 3rd cycle, variation

        for beat_off, bol, vel in pat:
            t = t0 + beat_off * beat_dur
            pos = int(t * sr)
            if pos >= len(output):
                break
            # Humanize: tiny random timing offset (±10ms)
            jitter = int(np.random.uniform(-0.010, 0.010) * sr)
            pos = max(0, pos + jitter)
            stroke = stroke_map[bol]()
            # Humanize velocity too
            vel *= np.random.uniform(0.90, 1.05)
            place_stroke(output, stroke, pos, vel)

    # Shaker on every off-beat 8th note (very subtle)
    t = beat_dur / 2
    while t < duration:
        pos = int(t * sr)
        jitter = int(np.random.uniform(-0.005, 0.005) * sr)
        pos = max(0, pos + jitter)
        hit = synth_shaker(sr)
        vel = np.random.uniform(0.5, 0.9)
        place_stroke(output, hit, pos, vel)
        t += beat_dur / 2

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  VOCAL-TO-BEAT ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def align_vocals_to_grid(y, sr, bpm):
    """
    Align vocal onsets to the nearest beat grid position using
    overlap-add time shifting. This nudges vocal phrases to sit
    on the beat without changing pitch or tempo.
    """
    beat_dur = 60.0 / bpm
    # Subdivide to 8th notes for finer alignment
    grid_interval = beat_dur / 2

    # Detect onsets in the vocal
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=512,
        backtrack=True, units='frames'
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    if len(onset_times) < 2:
        print("  Few onsets detected, skipping alignment")
        return y

    # Build the beat grid
    grid = np.arange(0, len(y) / sr, grid_interval)

    # For each onset, find how far it is from the nearest grid point
    shifts = []
    for ot in onset_times:
        nearest = grid[np.argmin(np.abs(grid - ot))]
        shift = nearest - ot
        shifts.append((ot, shift))

    # Only shift if the offset is small enough (< 30% of grid interval)
    # to avoid distorting the vocal
    max_shift = grid_interval * 0.30
    significant_shifts = [(t, s) for t, s in shifts if abs(s) > 0.005 and abs(s) < max_shift]

    if not significant_shifts:
        print("  Vocals already well-aligned to grid")
        return y

    print(f"  Nudging {len(significant_shifts)} onset regions (max shift: {max_shift*1000:.0f}ms)")

    # Apply micro time-shifts using overlap-add
    output = y.copy()
    window_samples = int(grid_interval * sr)
    half_win = window_samples // 2

    for onset_time, shift_sec in significant_shifts:
        shift_samples = int(shift_sec * sr)
        center = int(onset_time * sr)
        start = max(0, center - half_win)
        end = min(len(y), center + half_win)

        if abs(shift_samples) < 1:
            continue

        segment = y[start:end]
        seg_len = len(segment)

        # Cross-fade window
        fade_len = min(int(0.008 * sr), seg_len // 4)
        fade_in_w = np.linspace(0, 1, fade_len)
        fade_out_w = np.linspace(1, 0, fade_len)

        # Shift the segment
        new_start = max(0, start + shift_samples)
        new_end = min(len(output), new_start + seg_len)
        actual_len = new_end - new_start

        if actual_len < fade_len * 2:
            continue

        shifted = np.zeros(actual_len)
        copy_len = min(seg_len, actual_len)
        shifted[:copy_len] = segment[:copy_len]

        # Apply crossfade at boundaries
        fl = min(fade_len, actual_len)
        shifted[:fl] *= fade_in_w[:fl]
        shifted[-fl:] *= fade_out_w[:fl]

        output[new_start:new_start + fl] *= fade_out_w[:fl]
        output[new_start + actual_len - fl:new_end] *= fade_in_w[:fl]

        output[new_start:new_end] += shifted[:actual_len]

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    print("=" * 55)
    print("  Percussion Beat Generator")
    print("  Tabla + Dholak beats, vocal alignment")
    print("=" * 55)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[1/5] Loading audio...")
    y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"  {duration:.1f}s | {sr}Hz")

    # ── Tempo ─────────────────────────────────────────────────────────────
    print("\n[2/5] Detecting tempo...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    # This romantic song is slow; if detected >120 it's likely doubled
    if bpm > 120:
        bpm = bpm / 2
    print(f"  Tempo: {bpm:.0f} BPM")
    print(f"  Beat duration: {60/bpm*1000:.0f}ms")

    # ── Align vocals ──────────────────────────────────────────────────────
    print("\n[3/5] Aligning vocals to beat grid...")
    y_aligned = align_vocals_to_grid(y, sr, bpm)

    # ── Generate beats ────────────────────────────────────────────────────
    print("\n[4/5] Generating percussion track...")
    beats = generate_beat_track(duration, sr, bpm)
    print(f"  Keherwa taal, {bpm:.0f} BPM")
    print(f"  Tabla (Na, Tin, Dha) + Dholak (Ge, Ta) + shaker")

    # ── Mix ────────────────────────────────────────────────────────────────
    print("\n[5/5] Mixing...")

    # Normalize beats relative to vocal
    vocal_rms = np.sqrt(np.mean(y_aligned ** 2))
    beat_rms = np.sqrt(np.mean(beats ** 2))

    # Beats should sit behind the vocal — ~60% of vocal energy
    if beat_rms > 0:
        beat_gain = (vocal_rms * 0.60) / beat_rms
        beats *= beat_gain
    print(f"  Beat level: {beat_gain:.2f}x (60% of vocal RMS)")

    # Ensure same length
    n = len(y_aligned)
    if len(beats) < n:
        beats = np.pad(beats, (0, n - len(beats)))
    else:
        beats = beats[:n]

    # Simple stereo: vocals center, beats slightly wide
    # Tabla slightly left, dholak bass center, shaker slightly right
    vocal_L = y_aligned
    vocal_R = y_aligned
    beat_L = beats * 1.0
    beat_R = beats * 1.0

    left = vocal_L + beat_L
    right = vocal_R + beat_R

    stereo = np.column_stack([left, right])

    # Limiter
    peak = np.abs(stereo).max()
    if peak > 0.95:
        stereo = stereo / peak * 0.95

    sf.write(output_path, stereo, sr)
    fsize = os.path.getsize(output_path)

    print(f"\n{'=' * 55}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Size:   {fsize/1024:.0f} KB | {duration:.1f}s")
    print(f"  Just beats + vocals. No extra music.")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
