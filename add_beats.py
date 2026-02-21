#!/usr/bin/env python3
"""
Add percussion beats to a vocal recording using REAL drum samples (CC0).
Uses bongo, darbuka, and frame drum samples from VCSL, processed to
approximate tabla/dholak timbres. No extra music — just beats.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
import os
import glob as globmod

INPUT_FILE = "Mummy_Tumse_Milkar.m4a"
OUTPUT_FILE = "Mummy_Tumse_Milkar_with_beats.wav"
SAMPLE_RATE = 44100
SAMPLES_DIR = "samples"


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLE LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_sample(path, sr):
    """Load a WAV sample and convert to mono at target sample rate."""
    y, orig_sr = sf.read(path, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y


def pitch_shift_sample(y, sr, semitones):
    """Pitch shift a sample by n semitones."""
    if abs(semitones) < 0.01:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)


def add_resonance(y, sr, freq, q=15, gain=3.0):
    """Add a resonant peak at a frequency to give tabla-like ring."""
    nyq = sr / 2
    if freq >= nyq:
        return y
    w0 = freq / nyq
    b, a = signal.iirpeak(w0, q)
    filtered = signal.lfilter(b, a, y) * gain
    return y + filtered


def shape_decay(y, sr, decay_time=0.3):
    """Apply exponential decay envelope to tighten the sample."""
    n = len(y)
    t = np.arange(n) / sr
    env = np.exp(-t / decay_time)
    return y * env


def boost_low(y, sr, cutoff=200, gain=2.0):
    """Boost low frequencies for bass drum feel."""
    nyq = sr / 2
    b, a = signal.butter(2, cutoff / nyq, btype='low')
    low = signal.lfilter(b, a, y)
    return y + low * (gain - 1.0)


def add_pitch_bend_down(y, sr, bend_semitones=3, duration=0.15):
    """Add downward pitch bend at the start (bayan characteristic)."""
    bend_samples = int(duration * sr)
    if bend_samples >= len(y):
        bend_samples = len(y) - 1
    # Process in small chunks with varying pitch
    chunk_size = int(0.01 * sr)  # 10ms chunks
    output = y.copy()
    for i in range(0, bend_samples, chunk_size):
        end = min(i + chunk_size, bend_samples)
        progress = i / bend_samples  # 0 to 1
        # Pitch starts high and drops to normal
        current_shift = bend_semitones * (1.0 - progress)
        if current_shift > 0.1:
            chunk = y[i:end]
            # Simple resampling for pitch shift
            ratio = 2.0 ** (current_shift / 12.0)
            indices = np.arange(len(chunk)) * ratio
            indices = indices[indices < len(chunk)].astype(int)
            if len(indices) > 0:
                stretched = chunk[indices]
                fit_len = min(len(stretched), end - i)
                output[i:i+fit_len] = stretched[:fit_len]
    return output


class SampleKit:
    """Load and process drum samples into tabla-like voices."""

    def __init__(self, samples_dir, sr):
        self.sr = sr
        self.samples_dir = samples_dir
        self.voices = {}
        self._load_and_process()

    def _find(self, pattern):
        """Find sample files matching a pattern."""
        matches = sorted(globmod.glob(os.path.join(self.samples_dir, pattern)))
        return matches

    def _load_and_process(self):
        sr = self.sr

        # ── NA (treble ring) ← Bongo High open hits, pitched down ─────
        na_files = self._find("BongoH_Hit1_v*_rr*_Mid.wav")
        na_samples = []
        for f in na_files:
            y = load_sample(f, sr)
            # Pitch down ~3 semitones to lower into tabla range
            y = pitch_shift_sample(y, sr, -3)
            # Add resonance at ~350Hz (tabla Na characteristic ring)
            y = add_resonance(y, sr, freq=350, q=20, gain=2.0)
            # Shape decay — tabla Na rings for about 200ms
            y = shape_decay(y, sr, decay_time=0.20)
            # Normalize
            peak = np.abs(y).max()
            if peak > 0:
                y = y / peak * 0.85
            na_samples.append(y)
        self.voices['na'] = na_samples if na_samples else None
        print(f"    Na: {len(na_samples)} variations")

        # ── TIN (muted tap) ← Bongo High muted hits ───────────────────
        tin_files = self._find("BongoH_HitMuted1_v*_rr*_Mid.wav")
        tin_samples = []
        for f in tin_files:
            y = load_sample(f, sr)
            y = pitch_shift_sample(y, sr, -2)
            # Tighter decay for muted stroke
            y = shape_decay(y, sr, decay_time=0.08)
            peak = np.abs(y).max()
            if peak > 0:
                y = y / peak * 0.75
            tin_samples.append(y)
        self.voices['tin'] = tin_samples if tin_samples else None
        print(f"    Tin: {len(tin_samples)} variations")

        # ── GE (bass) ← Bongo Low open hits + low boost + pitch bend ──
        ge_files = self._find("BongoL_Hit1_v*_rr*_Mid.wav")
        ge_samples = []
        for f in ge_files:
            y = load_sample(f, sr)
            # Pitch way down for deep bass
            y = pitch_shift_sample(y, sr, -7)
            # Boost lows
            y = boost_low(y, sr, cutoff=150, gain=2.5)
            # Add pitch bend down (bayan characteristic)
            y = add_pitch_bend_down(y, sr, bend_semitones=2, duration=0.12)
            # Longer decay for bass
            y = shape_decay(y, sr, decay_time=0.30)
            peak = np.abs(y).max()
            if peak > 0:
                y = y / peak * 0.90
            ge_samples.append(y)
        self.voices['ge'] = ge_samples if ge_samples else None
        print(f"    Ge: {len(ge_samples)} variations")

        # ── DHA (composite) ← Frame Drum open hit + Darbuka bass ──────
        dha_hi_files = self._find("Darbuka_1_hit_vl*_rr*.wav")
        dha_lo_files = self._find("HDrumL_Hit_v*_rr*_Sum.wav")
        dha_samples = []
        for i in range(max(len(dha_hi_files), len(dha_lo_files))):
            hi_f = dha_hi_files[i % len(dha_hi_files)] if dha_hi_files else None
            lo_f = dha_lo_files[i % len(dha_lo_files)] if dha_lo_files else None
            if hi_f and lo_f:
                hi = load_sample(hi_f, sr)
                lo = load_sample(lo_f, sr)
                # Process treble (darbuka → dayan sound)
                hi = pitch_shift_sample(hi, sr, -2)
                hi = add_resonance(hi, sr, freq=380, q=18, gain=1.5)
                hi = shape_decay(hi, sr, decay_time=0.18)
                # Process bass (frame drum → bayan sound)
                lo = pitch_shift_sample(lo, sr, -5)
                lo = boost_low(lo, sr, cutoff=120, gain=2.0)
                lo = shape_decay(lo, sr, decay_time=0.28)
                # Combine
                n = max(len(hi), len(lo))
                combined = np.zeros(n)
                combined[:len(hi)] += hi * 0.5
                combined[:len(lo)] += lo * 0.65
                peak = np.abs(combined).max()
                if peak > 0:
                    combined = combined / peak * 0.90
                dha_samples.append(combined)
        self.voices['dha'] = dha_samples if dha_samples else None
        print(f"    Dha: {len(dha_samples)} variations")

        # ── TA (sharp slap) ← Darbuka sharp hits ─────────────────────
        ta_files = self._find("Darbuka_[3-5]_hit_vl*_rr*.wav")
        ta_samples = []
        for f in ta_files:
            y = load_sample(f, sr)
            y = shape_decay(y, sr, decay_time=0.10)
            peak = np.abs(y).max()
            if peak > 0:
                y = y / peak * 0.70
            ta_samples.append(y)
        self.voices['ta'] = ta_samples if ta_samples else None
        print(f"    Ta: {len(ta_samples)} variations")

        # ── GHOST (very soft brush) ← Frame drum muted, quiet ────────
        ghost_files = self._find("HDrumS_HitMuted_v*_rr*_Sum.wav")
        ghost_samples = []
        for f in ghost_files:
            y = load_sample(f, sr)
            y = shape_decay(y, sr, decay_time=0.06)
            peak = np.abs(y).max()
            if peak > 0:
                y = y / peak * 0.30
            ghost_samples.append(y)
        self.voices['ghost'] = ghost_samples if ghost_samples else None
        print(f"    Ghost: {len(ghost_samples)} variations")

    def get(self, voice_name):
        """Get a random variation of a voice. Returns None if unavailable."""
        samples = self.voices.get(voice_name)
        if not samples:
            return None
        return samples[np.random.randint(len(samples))].copy()


# ═══════════════════════════════════════════════════════════════════════════════
#  BEAT PATTERN GENERATOR (using real samples)
# ═══════════════════════════════════════════════════════════════════════════════

def place_stroke(output, stroke, pos, velocity=1.0):
    """Place a percussion stroke into the output buffer."""
    if stroke is None:
        return
    end = min(pos + len(stroke), len(output))
    if 0 <= pos < len(output):
        output[pos:end] += stroke[:end - pos] * velocity


def generate_beat_track(kit, duration, sr, bpm):
    """
    Keherwa taal using real processed samples.

    Beat structure (8-beat cycle):
      1: Dha (accent - sam)
      2: Ge
      3: Na
      4: Ti
      5: Na
      6: Ka (ghost)
      7: Dha
      8: Na

    With ghost notes on off-beats for groove.
    """
    beat_dur = 60.0 / bpm
    output = np.zeros(int(duration * sr))

    # Main pattern
    pat_a = [
        (0,    'dha',   0.95),
        (1,    'ge',    0.72),
        (2,    'na',    0.82),
        (3,    'tin',   0.55),
        (4,    'na',    0.78),
        (5,    'ghost', 0.35),
        (6,    'dha',   0.88),
        (7,    'na',    0.65),
    ]
    # Variation with 16th-note fills
    pat_b = [
        (0,    'dha',   0.95),
        (1,    'ge',    0.72),
        (2,    'na',    0.78),
        (2.5,  'tin',   0.38),
        (3,    'na',    0.58),
        (4,    'dha',   0.85),
        (5,    'ghost', 0.30),
        (5.5,  'tin',   0.35),
        (6,    'na',    0.78),
        (6.5,  'ghost', 0.28),
        (7,    'na',    0.58),
    ]
    # Fill pattern (pre-sam tihai-like)
    pat_fill = [
        (0,    'dha',   0.95),
        (1,    'ge',    0.70),
        (2,    'ta',    0.72),
        (2.5,  'tin',   0.55),
        (3,    'ta',    0.65),
        (3.5,  'tin',   0.50),
        (4,    'na',    0.80),
        (4.5,  'ghost', 0.35),
        (5,    'ta',    0.68),
        (5.5,  'tin',   0.52),
        (6,    'ta',    0.72),
        (6.5,  'tin',   0.55),
        (7,    'dha',   0.80),
        (7.5,  'ghost', 0.30),
    ]

    cycle_dur = 8 * beat_dur
    n_cycles = int(duration / cycle_dur) + 1

    for cyc in range(n_cycles):
        t0 = cyc * cycle_dur

        # Pick pattern: mostly A, every 3rd B, every 4th fill
        if cyc % 4 == 3:
            pat = pat_fill
        elif cyc % 3 == 2:
            pat = pat_b
        else:
            pat = pat_a

        for beat_off, bol, vel in pat:
            t = t0 + beat_off * beat_dur
            pos = int(t * sr)
            if pos >= len(output):
                break

            # Humanize timing: ±8ms jitter
            jitter = int(np.random.uniform(-0.008, 0.008) * sr)
            pos = max(0, pos + jitter)

            stroke = kit.get(bol)
            # Humanize velocity
            vel *= np.random.uniform(0.92, 1.06)
            place_stroke(output, stroke, pos, vel)

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  VOCAL-TO-BEAT ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def align_vocals_to_grid(y, sr, bpm):
    """
    Gently nudge vocal onsets toward the nearest beat grid position.
    Uses overlap-add with crossfades to avoid artifacts.
    Only shifts onsets that are slightly off — preserves natural phrasing.
    """
    beat_dur = 60.0 / bpm
    grid_interval = beat_dur / 2  # 8th note grid

    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=512, backtrack=True, units='frames'
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    if len(onset_times) < 2:
        print("  Few onsets detected, skipping alignment")
        return y

    grid = np.arange(0, len(y) / sr, grid_interval)

    shifts = []
    for ot in onset_times:
        nearest = grid[np.argmin(np.abs(grid - ot))]
        shift = nearest - ot
        shifts.append((ot, shift))

    # Only nudge if offset is small (< 25% of grid) to preserve natural feel
    max_shift = grid_interval * 0.25
    significant = [(t, s) for t, s in shifts if 0.005 < abs(s) < max_shift]

    if not significant:
        print("  Vocals already well-aligned to grid")
        return y

    print(f"  Nudging {len(significant)} onset regions (max shift: {max_shift*1000:.0f}ms)")

    output = y.copy()
    window_samples = int(grid_interval * sr)
    half_win = window_samples // 2

    for onset_time, shift_sec in significant:
        shift_samples = int(shift_sec * sr)
        center = int(onset_time * sr)
        start = max(0, center - half_win)
        end = min(len(y), center + half_win)

        if abs(shift_samples) < 1:
            continue

        segment = y[start:end]
        seg_len = len(segment)
        fade_len = min(int(0.008 * sr), seg_len // 4)
        fade_in_w = np.linspace(0, 1, fade_len)
        fade_out_w = np.linspace(1, 0, fade_len)

        new_start = max(0, start + shift_samples)
        new_end = min(len(output), new_start + seg_len)
        actual_len = new_end - new_start

        if actual_len < fade_len * 2:
            continue

        shifted = np.zeros(actual_len)
        copy_len = min(seg_len, actual_len)
        shifted[:copy_len] = segment[:copy_len]

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
    samples_path = os.path.join(script_dir, SAMPLES_DIR)

    print("=" * 60)
    print("  Percussion Beat Generator (Real Samples)")
    print("  CC0 drum samples processed into tabla-like timbres")
    print("=" * 60)

    # ── Load vocal ────────────────────────────────────────────────────────
    print("\n[1/6] Loading vocal...")
    y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"  {duration:.1f}s | {sr}Hz")

    # ── Tempo ─────────────────────────────────────────────────────────────
    print("\n[2/6] Detecting tempo...")
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    if bpm > 120:
        bpm = bpm / 2
    print(f"  Tempo: {bpm:.0f} BPM")

    # ── Load & process samples ────────────────────────────────────────────
    print("\n[3/6] Loading & processing drum samples...")
    kit = SampleKit(samples_path, sr)

    # ── Align vocals ──────────────────────────────────────────────────────
    print("\n[4/6] Aligning vocals to beat grid...")
    y_aligned = align_vocals_to_grid(y, sr, bpm)

    # ── Generate beats ────────────────────────────────────────────────────
    print("\n[5/6] Generating beat track...")
    beats = generate_beat_track(kit, duration, sr, bpm)
    print(f"  Keherwa taal at {bpm:.0f} BPM")
    print(f"  Voices: Dha, Na, Ge, Tin, Ta, Ghost")

    # ── Mix ────────────────────────────────────────────────────────────────
    print("\n[6/6] Mixing...")

    # Level: beats at ~55% of vocal RMS energy
    vocal_rms = np.sqrt(np.mean(y_aligned ** 2))
    beat_rms = np.sqrt(np.mean(beats ** 2))
    beat_gain = 1.0
    if beat_rms > 0:
        beat_gain = (vocal_rms * 0.55) / beat_rms
        beats *= beat_gain
    print(f"  Beat gain: {beat_gain:.2f}x")

    # Match lengths
    n = len(y_aligned)
    if len(beats) < n:
        beats = np.pad(beats, (0, n - len(beats)))
    else:
        beats = beats[:n]

    # Stereo mix: vocals center, beats center
    left = y_aligned + beats
    right = y_aligned + beats

    stereo = np.column_stack([left, right])

    # Soft limiter
    peak = np.abs(stereo).max()
    if peak > 0.95:
        stereo = stereo / peak * 0.95

    sf.write(output_path, stereo, sr)
    fsize = os.path.getsize(output_path)

    print(f"\n{'=' * 60}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Size:   {fsize/1024:.0f} KB | {duration:.1f}s")
    print(f"  Real drum samples — just beats, no extra music.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
