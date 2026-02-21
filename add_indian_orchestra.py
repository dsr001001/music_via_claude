#!/usr/bin/env python3
"""
Indian Classical Orchestra Accompaniment Generator
Adds tabla, bansuri, santoor, sitar, and string ensemble to a vocal recording.
Arrangement style: tasteful, with instruments entering gradually and
melodic fills placed in vocal gaps.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
import os

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE = "Mummy_Tumse_Milkar.m4a"
OUTPUT_FILE = "Mummy_Tumse_Milkar_with_orchestra.wav"
SAMPLE_RATE = 48000

VOCAL_LEVEL = 1.0
ORCHESTRA_LEVEL = 0.42

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ═══════════════════════════════════════════════════════════════════════════════
#  CHORD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

CHORD_TEMPLATES = {}
for root in range(12):
    major = np.zeros(12)
    major[root] = 1.0; major[(root+4)%12] = 0.8; major[(root+7)%12] = 0.8
    CHORD_TEMPLATES[f"{NOTE_NAMES[root]}"] = major
    minor = np.zeros(12)
    minor[root] = 1.0; minor[(root+3)%12] = 0.8; minor[(root+7)%12] = 0.8
    CHORD_TEMPLATES[f"{NOTE_NAMES[root]}m"] = minor
    sev = np.zeros(12)
    sev[root] = 1.0; sev[(root+4)%12] = 0.7; sev[(root+7)%12] = 0.7; sev[(root+10)%12] = 0.6
    CHORD_TEMPLATES[f"{NOTE_NAMES[root]}7"] = sev
    m7 = np.zeros(12)
    m7[root] = 1.0; m7[(root+3)%12] = 0.7; m7[(root+7)%12] = 0.7; m7[(root+10)%12] = 0.6
    CHORD_TEMPLATES[f"{NOTE_NAMES[root]}m7"] = m7


def detect_chords(y, sr, hop_length=512, segment_duration=0.5):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_smooth = uniform_filter1d(chroma, size=10, axis=1)
    segment_frames = int(segment_duration * sr / hop_length)
    n_segments = chroma_smooth.shape[1] // segment_frames
    chords = []
    for seg in range(n_segments):
        start = seg * segment_frames
        end = start + segment_frames
        seg_chroma = chroma_smooth[:, start:end].mean(axis=1)
        norm = np.linalg.norm(seg_chroma)
        if norm > 0:
            seg_chroma = seg_chroma / norm
        best_chord, best_score = "C", -1
        for name, template in CHORD_TEMPLATES.items():
            t_norm = template / np.linalg.norm(template)
            score = np.dot(seg_chroma, t_norm)
            if score > best_score:
                best_score = score
                best_chord = name
        chords.append((seg * segment_duration, segment_duration, best_chord, best_score))
    return chords


def smooth_chord_progression(chords, min_duration=1.5):
    if not chords:
        return chords
    smoothed = [chords[0]]
    acc_dur = chords[0][1]
    for i in range(1, len(chords)):
        time, dur, chord, score = chords[i]
        prev = smoothed[-1][2]
        if chord == prev:
            old = smoothed[-1]
            smoothed[-1] = (old[0], old[1] + dur, old[2], max(old[3], score))
            acc_dur += dur
        elif acc_dur >= min_duration:
            smoothed.append((time, dur, chord, score))
            acc_dur = dur
        else:
            old = smoothed[-1]
            smoothed[-1] = (old[0], old[1] + dur, old[2], old[3])
            acc_dur += dur
    return smoothed


def chord_to_semitones(chord_name):
    if chord_name.endswith('m7'):
        root_name, intervals = chord_name[:-2], [0, 3, 7, 10]
    elif chord_name.endswith('7'):
        root_name, intervals = chord_name[:-1], [0, 4, 7, 10]
    elif chord_name.endswith('m'):
        root_name, intervals = chord_name[:-1], [0, 3, 7]
    else:
        root_name, intervals = chord_name, [0, 4, 7]
    root = NOTE_NAMES.index(root_name)
    return root, [(root + iv) % 12 for iv in intervals]


def semitone_to_freq(semitone, octave):
    midi = semitone + (octave + 1) * 12
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  VOCAL ACTIVITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_vocal_gaps(y, sr, threshold_factor=0.3, min_gap_duration=0.4):
    """Detect gaps in vocal activity where fills can be placed."""
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_smooth = uniform_filter1d(rms, size=20)
    threshold = rms_smooth.max() * threshold_factor

    times = librosa.frames_to_time(np.arange(len(rms_smooth)), sr=sr, hop_length=hop)
    is_silent = rms_smooth < threshold

    gaps = []
    in_gap = False
    gap_start = 0.0
    for i, silent in enumerate(is_silent):
        if silent and not in_gap:
            in_gap = True
            gap_start = times[i]
        elif not silent and in_gap:
            in_gap = False
            gap_end = times[i]
            if gap_end - gap_start >= min_gap_duration:
                gaps.append((gap_start, gap_end))
    if in_gap:
        gap_end = times[-1]
        if gap_end - gap_start >= min_gap_duration:
            gaps.append((gap_start, gap_end))
    return gaps


def compute_dynamics_envelope(y, sr, hop_length=512, smooth_window=1.5):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    smooth_frames = int(smooth_window * sr / hop_length)
    rms_smooth = uniform_filter1d(rms, size=max(smooth_frames, 1))
    max_rms = rms_smooth.max()
    if max_rms > 0:
        rms_smooth = rms_smooth / max_rms
    times = librosa.frames_to_time(np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length)
    t_samples = np.arange(len(y)) / sr
    envelope = np.interp(t_samples, times, rms_smooth)
    envelope = np.clip(envelope, 0.1, 1.0)
    envelope = 0.25 + 0.75 * envelope
    return envelope


# ═══════════════════════════════════════════════════════════════════════════════
#  INSTRUMENT SYNTHESIZERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Tabla ─────────────────────────────────────────────────────────────────────

def synth_tabla_na(sr, pitch=380, duration=0.15):
    """Dayan stroke: Na/Tin - sharp, high-pitched ring."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Decaying sine with slight pitch drop
    freq = pitch * np.exp(-t * 2.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    tone = np.sin(phase) * 0.7
    # Add harmonics for timbral richness
    tone += np.sin(phase * 2.03) * 0.2 * np.exp(-t * 15)
    tone += np.sin(phase * 3.01) * 0.1 * np.exp(-t * 20)
    # Sharp attack noise (slap)
    noise = np.random.randn(n) * 0.15 * np.exp(-t * 80)
    b, a = signal.butter(4, min(pitch * 3, sr * 0.45) / (sr/2), btype='low')
    noise = signal.lfilter(b, a, noise)
    # Envelope
    env = np.exp(-t * 12)
    env[:int(0.002 * sr)] = np.linspace(0, 1, int(0.002 * sr))
    return (tone + noise) * env


def synth_tabla_ge(sr, pitch=90, duration=0.3):
    """Bayan stroke: Ge/Ghe - deep bass with pitch glide."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Pitch glides down characteristically
    freq = pitch * np.exp(-t * 3.0) + 40
    phase = 2 * np.pi * np.cumsum(freq) / sr
    tone = np.sin(phase) * 0.9
    tone += np.sin(phase * 1.5) * 0.15 * np.exp(-t * 8)
    # Envelope: slower decay for bass
    env = np.exp(-t * 5)
    env[:int(0.005 * sr)] = np.linspace(0, 1, int(0.005 * sr))
    return tone * env


def synth_tabla_dha(sr, na_pitch=380, ge_pitch=90):
    """Dha = Na + Ge together (open stroke)."""
    na = synth_tabla_na(sr, na_pitch, duration=0.18)
    ge = synth_tabla_ge(sr, ge_pitch, duration=0.3)
    n = max(len(na), len(ge))
    out = np.zeros(n)
    out[:len(na)] += na * 0.6
    out[:len(ge)] += ge * 0.7
    return out


def synth_tabla_ti(sr, pitch=500, duration=0.08):
    """Ti - short, high tap."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * pitch * t) * 0.4
    noise = np.random.randn(n) * 0.1 * np.exp(-t * 100)
    env = np.exp(-t * 30)
    env[:int(0.001 * sr)] = np.linspace(0, 1, int(0.001 * sr))
    return (tone + noise) * env


def generate_tabla_pattern(duration, sr, tempo_bpm):
    """Generate a Keherwa-based tabla pattern for the full duration."""
    beat_dur = 60.0 / tempo_bpm
    output = np.zeros(int(duration * sr))

    # Keherwa taal (8 beats): Dha Ge Na Ti Na Ka Dhi Na
    # Simplified pattern with variations
    pattern_a = [
        (0.0,   'dha'),
        (1.0,   'ge'),
        (2.0,   'na'),
        (3.0,   'ti'),
        (4.0,   'na'),
        (5.0,   'ti'),
        (6.0,   'dha'),
        (7.0,   'na'),
    ]
    pattern_b = [
        (0.0,   'dha'),
        (1.0,   'ge'),
        (2.0,   'na'),
        (2.5,   'ti'),
        (3.0,   'na'),
        (4.0,   'dha'),
        (5.0,   'ti'),
        (6.0,   'na'),
        (6.5,   'ti'),
        (7.0,   'na'),
    ]

    cycle_dur = 8 * beat_dur
    n_cycles = int(duration / cycle_dur) + 1

    for cycle in range(n_cycles):
        cycle_start = cycle * cycle_dur
        # Alternate patterns for variation
        pattern = pattern_a if cycle % 2 == 0 else pattern_b

        for beat_offset, bol in pattern:
            t = cycle_start + beat_offset * beat_dur
            sample_pos = int(t * sr)
            if sample_pos >= len(output):
                break

            # Synthesize the bol
            if bol == 'dha':
                stroke = synth_tabla_dha(sr)
            elif bol == 'ge':
                stroke = synth_tabla_ge(sr)
            elif bol == 'na':
                stroke = synth_tabla_na(sr)
            elif bol == 'ti':
                stroke = synth_tabla_ti(sr)
            else:
                continue

            end = min(sample_pos + len(stroke), len(output))
            output[sample_pos:end] += stroke[:end - sample_pos]

    return output


# ── Shaker / Hi-hat ───────────────────────────────────────────────────────────

def synth_shaker_hit(sr, duration=0.05):
    """Single shaker/hi-hat hit."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    noise = np.random.randn(n)
    # Bandpass filter for metallic character
    nyq = sr / 2
    b, a = signal.butter(3, [4000/nyq, min(12000, nyq*0.95)/nyq], btype='band')
    noise = signal.lfilter(b, a, noise)
    env = np.exp(-t * 40)
    env[:int(0.001*sr)] = np.linspace(0, 1, int(0.001*sr))
    return noise * env * 0.15


def generate_shaker_pattern(duration, sr, tempo_bpm):
    """Light shaker on off-beats (8th notes)."""
    beat_dur = 60.0 / tempo_bpm
    eighth_dur = beat_dur / 2
    output = np.zeros(int(duration * sr))

    t = eighth_dur  # start on the "and" of beat 1
    while t < duration:
        pos = int(t * sr)
        hit = synth_shaker_hit(sr)
        end = min(pos + len(hit), len(output))
        if pos < len(output):
            # Vary velocity slightly for humanization
            vel = np.random.uniform(0.6, 1.0)
            output[pos:end] += hit[:end-pos] * vel
        t += eighth_dur

    return output


# ── Strings (Violins) ────────────────────────────────────────────────────────

def synth_string_tone(freq, duration, sr, brightness=0.3, vibrato_rate=5.5,
                      vibrato_depth=0.003, n_voices=3, attack=0.15, release=0.15):
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Envelope
    att_n = int(attack * sr)
    rel_n = int(release * sr)
    env = np.ones(n)
    if att_n > 0 and att_n < n:
        env[:att_n] = np.linspace(0, 1, att_n)
    if rel_n > 0 and rel_n < n:
        env[-rel_n:] = np.linspace(1, 0, rel_n)
    # Vibrato
    vib = vibrato_depth * freq * np.sin(2 * np.pi * vibrato_rate * t)
    out = np.zeros(n)
    detunes = np.linspace(-8, 8, n_voices) if n_voices > 1 else [0]
    for det in detunes:
        vf = freq * (2.0 ** (det / 1200.0))
        phase = np.cumsum(2 * np.pi * (vf + vib) / sr)
        saw = signal.sawtooth(phase)
        sine = np.sin(phase)
        out += ((1 - brightness) * sine + brightness * saw) / n_voices
    # Low-pass filter
    cutoff = min(freq * 5, sr * 0.45)
    b, a = signal.butter(3, cutoff / (sr/2), btype='low')
    out = signal.filtfilt(b, a, out)
    return out * env


def generate_strings(chords, duration, sr):
    """Generate string pad following chord progression."""
    output = np.zeros(int(duration * sr))
    for time_start, dur, chord, _ in chords:
        _, notes = chord_to_semitones(chord)
        start = int(time_start * sr)
        seg_dur = min(dur + 0.15, duration - time_start)  # slight overlap
        # Play chord tones in violin range (octave 3-4)
        for i, note in enumerate(notes[:3]):
            octave = 4 if i == 0 else 3
            freq = semitone_to_freq(note, octave)
            tone = synth_string_tone(freq, seg_dur, sr, brightness=0.25,
                                     attack=0.2, release=0.2, n_voices=3)
            end = min(start + len(tone), len(output))
            output[start:end] += tone[:end-start] * 0.35
        # Cello: root an octave lower
        freq = semitone_to_freq(notes[0], 2)
        tone = synth_string_tone(freq, seg_dur, sr, brightness=0.2,
                                 attack=0.25, release=0.25, n_voices=2,
                                 vibrato_depth=0.002)
        end = min(start + len(tone), len(output))
        output[start:end] += tone[:end-start] * 0.4
    return output


# ── Bansuri (Bamboo Flute) ────────────────────────────────────────────────────

def synth_bansuri_note(freq, duration, sr, breath_amount=0.12):
    """Synthesize a bansuri-like tone with breath noise and wide vibrato."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Slow attack (breath start), sustain, gentle release
    att = int(0.12 * sr)
    rel = int(0.15 * sr)
    env = np.ones(n)
    if att < n:
        env[:att] = np.linspace(0, 1, att) ** 1.5  # gentle curve
    if rel < n:
        env[-rel:] = np.linspace(1, 0, rel) ** 1.5
    # Wide vibrato characteristic of bansuri
    vib_rate = 5.0 + np.random.uniform(-0.5, 0.5)
    vib_depth = 0.008 * freq  # wider than violin
    # Delayed vibrato onset
    vib_env = np.clip(t / 0.3, 0, 1)  # vibrato fades in over 0.3s
    vib = vib_depth * np.sin(2 * np.pi * vib_rate * t) * vib_env
    phase = np.cumsum(2 * np.pi * (freq + vib) / sr)
    # Flute = mostly fundamental + soft overtones
    tone = np.sin(phase) * 0.8
    tone += np.sin(phase * 2) * 0.15  # octave
    tone += np.sin(phase * 3) * 0.05  # twelfth
    # Breath noise (bandpass around the pitch)
    noise = np.random.randn(n) * breath_amount
    nyq = sr / 2
    lo = max(freq * 0.5, 100) / nyq
    hi = min(freq * 4, nyq * 0.9) / nyq
    if lo < hi < 1:
        b, a = signal.butter(2, [lo, hi], btype='band')
        noise = signal.lfilter(b, a, noise)
    tone += noise
    return tone * env


def generate_bansuri_fills(chords, gaps, duration, sr):
    """Play bansuri melodic fills during vocal gaps."""
    output = np.zeros(int(duration * sr))

    # Build a time->chord lookup
    def chord_at_time(t):
        for cs, cd, cn, _ in chords:
            if cs <= t < cs + cd:
                return cn
        return chords[-1][2] if chords else "F"

    for gap_start, gap_end in gaps:
        gap_dur = gap_end - gap_start
        if gap_dur < 0.5:
            continue  # too short for a fill

        chord = chord_at_time(gap_start)
        root, notes = chord_to_semitones(chord)

        # Create a small melodic phrase using chord tones + passing tones
        # Pentatonic scale based on root
        scale = [(root + s) % 12 for s in [0, 2, 4, 7, 9]]

        # Generate 2-5 notes that fit the gap
        n_notes = min(max(2, int(gap_dur / 0.35)), 5)
        note_dur = min(gap_dur / n_notes, 0.6)
        fill_start = gap_start + 0.05  # tiny offset

        # Melodic contour: arch shape (up then down)
        if n_notes <= 2:
            contour = [0, 1]
        elif n_notes == 3:
            contour = [0, 2, 1]
        elif n_notes == 4:
            contour = [0, 2, 3, 1]
        else:
            contour = [0, 1, 3, 2, 0]

        for i, ci in enumerate(contour[:n_notes]):
            note_time = fill_start + i * note_dur
            if note_time + note_dur > gap_end:
                break
            note_semi = scale[ci % len(scale)]
            freq = semitone_to_freq(note_semi, 5)  # bansuri octave 5

            this_dur = note_dur * np.random.uniform(0.85, 1.0)  # humanize
            tone = synth_bansuri_note(freq, this_dur, sr)
            pos = int(note_time * sr)
            end = min(pos + len(tone), len(output))
            if pos >= 0 and pos < len(output):
                output[pos:end] += tone[:end-pos] * 0.5

    return output


# ── Santoor ───────────────────────────────────────────────────────────────────

def synth_santoor_note(freq, duration, sr):
    """Hammered dulcimer-like tone: sharp attack, shimmering decay."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Very sharp attack
    env = np.exp(-t * 4.0)  # natural decay
    env[:int(0.002 * sr)] = np.linspace(0, 1, int(0.002 * sr))
    # Multiple harmonics (hammered string has rich spectrum)
    tone = np.sin(2 * np.pi * freq * t) * 0.5
    tone += np.sin(2 * np.pi * freq * 2.01 * t) * 0.25  # slight inharmonicity
    tone += np.sin(2 * np.pi * freq * 3.02 * t) * 0.12
    tone += np.sin(2 * np.pi * freq * 4.05 * t) * 0.06
    # Slight chorus for shimmer (2 detuned copies)
    tone2 = np.sin(2 * np.pi * freq * 1.002 * t) * 0.3
    tone2 += np.sin(2 * np.pi * freq * 0.998 * t) * 0.3
    return (tone + tone2 * 0.3) * env


def generate_santoor_runs(chords, duration, sr, tempo_bpm):
    """Occasional shimmering santoor arpeggios at key moments."""
    output = np.zeros(int(duration * sr))
    beat_dur = 60.0 / tempo_bpm

    # Place santoor runs every ~4-6 seconds (sparse for tasteful arrangement)
    run_interval = 4.5
    t = 3.0  # start after intro

    while t < duration - 1.0:
        # Find chord at this time
        chord = "F"
        for cs, cd, cn, _ in chords:
            if cs <= t < cs + cd:
                chord = cn
                break

        root, notes = chord_to_semitones(chord)
        scale = [(root + s) % 12 for s in [0, 2, 4, 7, 9]]  # pentatonic

        # Arpeggio: 4-6 rapid notes ascending
        n_notes = np.random.choice([4, 5, 6])
        note_spacing = beat_dur * 0.25  # 16th notes

        for i in range(n_notes):
            note_time = t + i * note_spacing
            if note_time >= duration:
                break
            note_semi = scale[i % len(scale)]
            # Ascending octaves 4-5
            octave = 4 + (i // len(scale))
            freq = semitone_to_freq(note_semi, min(octave, 5))
            tone = synth_santoor_note(freq, 0.4, sr)
            pos = int(note_time * sr)
            end = min(pos + len(tone), len(output))
            if pos >= 0 and pos < len(output):
                vel = 0.3 + 0.1 * (i / n_notes)  # crescendo
                output[pos:end] += tone[:end-pos] * vel

        t += run_interval + np.random.uniform(-0.5, 0.5)

    return output


# ── Sitar ─────────────────────────────────────────────────────────────────────

def synth_sitar_note(freq, duration, sr):
    """Plucked sitar tone with characteristic buzz (jawari)."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Pluck envelope: instant attack, slow decay
    env = np.exp(-t * 2.5)
    env[:int(0.001 * sr)] = np.linspace(0, 1, int(0.001 * sr))
    # Base tone
    phase = 2 * np.pi * freq * t
    tone = np.sin(phase) * 0.5
    # Rich harmonics (sitar has many)
    for h in range(2, 8):
        harm_amp = 0.3 / h
        # Slight inharmonicity
        tone += np.sin(phase * (h + np.random.uniform(-0.01, 0.01))) * harm_amp
    # Jawari buzz: soft clipping of the waveform
    tone = np.tanh(tone * 2.0) * 0.6
    # Sympathetic string resonance: faint high octave ring
    sympathetic = np.sin(phase * 2) * 0.08 * np.exp(-t * 1.5)
    return (tone + sympathetic) * env


def synth_sitar_meend(freq_start, freq_end, duration, sr):
    """Sitar meend (glide between notes)."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    # Smooth pitch glide
    freq = freq_start + (freq_end - freq_start) * (t / (duration + 1e-9))
    phase = np.cumsum(2 * np.pi * freq / sr)
    env = np.exp(-t * 2.0)
    tone = np.sin(phase) * 0.5
    for h in range(2, 6):
        tone += np.sin(phase * h) * (0.25 / h)
    tone = np.tanh(tone * 1.8) * 0.6
    return tone * env


def generate_sitar_phrases(chords, gaps, duration, sr):
    """Occasional sitar phrases for Indian classical flavor, placed in gaps."""
    output = np.zeros(int(duration * sr))

    def chord_at_time(t):
        for cs, cd, cn, _ in chords:
            if cs <= t < cs + cd:
                return cn
        return chords[-1][2]

    # Use every other gap (sparse) and some gaps for sitar
    sitar_gaps = [g for i, g in enumerate(gaps) if i % 3 == 1 and g[1]-g[0] > 0.6]

    # Also add a few timed placements if there aren't enough gaps
    if len(sitar_gaps) < 2:
        # Place at ~25% and ~75% of the track
        for t_frac in [0.25, 0.7]:
            t = t_frac * duration
            sitar_gaps.append((t, t + 0.8))

    for gap_start, gap_end in sitar_gaps:
        gap_dur = gap_end - gap_start
        chord = chord_at_time(gap_start)
        root, notes = chord_to_semitones(chord)

        # Short phrase: 2-3 notes with possible meend
        fill_start = gap_start + 0.05
        n_notes = min(3, max(1, int(gap_dur / 0.4)))

        scale = [(root + s) % 12 for s in [0, 2, 4, 7, 9]]

        for i in range(n_notes):
            note_time = fill_start + i * 0.35
            if note_time + 0.3 > gap_end:
                break
            note_semi = scale[i % len(scale)]
            freq = semitone_to_freq(note_semi, 4)  # sitar mid-range

            # Occasionally use meend (glide)
            if i > 0 and np.random.random() < 0.4:
                prev_semi = scale[(i-1) % len(scale)]
                prev_freq = semitone_to_freq(prev_semi, 4)
                tone = synth_sitar_meend(prev_freq, freq, 0.35, sr)
            else:
                tone = synth_sitar_note(freq, 0.5, sr)

            pos = int(note_time * sr)
            end = min(pos + len(tone), len(output))
            if pos >= 0 and pos < len(output):
                output[pos:end] += tone[:end-pos] * 0.3

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_reverb(audio, sr, decay=0.3, delay_ms=35, mix=0.2, n_taps=5):
    output = audio.copy()
    for tap in range(1, n_taps + 1):
        delay_samples = int(tap * delay_ms * sr / 1000)
        gain = decay ** tap
        if delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * gain
            output += delayed * mix
    return output


def fade_in(audio, sr, duration=2.0):
    """Apply a gradual fade-in to audio."""
    n = int(duration * sr)
    n = min(n, len(audio))
    audio[:n] *= np.linspace(0, 1, n) ** 1.5
    return audio


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    print("=" * 65)
    print("  Indian Classical Orchestra Accompaniment Generator")
    print("  Tabla + Bansuri + Santoor + Sitar + Strings")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[1/7] Loading audio...")
    y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"  {duration:.1f}s | {sr}Hz | {len(y)} samples")

    # ── Tempo ─────────────────────────────────────────────────────────────
    print("\n[2/7] Detecting tempo and beats...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    # The original song is moderate tempo; if detected >120, might be double
    if tempo_val > 120:
        effective_tempo = tempo_val / 2
    else:
        effective_tempo = tempo_val
    print(f"  Raw tempo: {tempo_val:.0f} BPM → Using: {effective_tempo:.0f} BPM")

    # ── Chords ────────────────────────────────────────────────────────────
    print("\n[3/7] Detecting chord progression...")
    raw_chords = detect_chords(y, sr)
    chords = smooth_chord_progression(raw_chords, min_duration=1.5)
    for cs, cd, cn, sc in chords:
        print(f"    {cs:5.1f}s - {cs+cd:5.1f}s: {cn:4s} ({sc:.2f})")

    # ── Vocal gaps ────────────────────────────────────────────────────────
    print("\n[4/7] Detecting vocal gaps for melodic fills...")
    gaps = detect_vocal_gaps(y, sr)
    print(f"  Found {len(gaps)} gaps:")
    for gs, ge in gaps:
        print(f"    {gs:.1f}s - {ge:.1f}s ({ge-gs:.1f}s)")

    # ── Dynamics ──────────────────────────────────────────────────────────
    print("\n[5/7] Computing dynamics envelope...")
    dynamics = compute_dynamics_envelope(y, sr)

    # ── Synthesize all instruments ────────────────────────────────────────
    print("\n[6/7] Synthesizing instruments...")

    print("  → Tabla...")
    tabla = generate_tabla_pattern(duration, sr, effective_tempo)
    tabla = apply_reverb(tabla, sr, decay=0.15, delay_ms=20, mix=0.1)

    print("  → Shaker...")
    shaker = generate_shaker_pattern(duration, sr, effective_tempo)

    print("  → Strings...")
    strings = generate_strings(chords, duration, sr)
    strings = apply_reverb(strings, sr, decay=0.3, delay_ms=40, mix=0.25)
    strings = fade_in(strings, sr, duration=2.5)  # strings enter gradually

    print("  → Bansuri...")
    bansuri = generate_bansuri_fills(chords, gaps, duration, sr)
    bansuri = apply_reverb(bansuri, sr, decay=0.35, delay_ms=50, mix=0.3)
    bansuri = fade_in(bansuri, sr, duration=4.0)  # bansuri enters later

    print("  → Santoor...")
    santoor = generate_santoor_runs(chords, duration, sr, effective_tempo)
    santoor = apply_reverb(santoor, sr, decay=0.25, delay_ms=30, mix=0.2)
    santoor = fade_in(santoor, sr, duration=5.0)  # santoor enters even later

    print("  → Sitar...")
    sitar = generate_sitar_phrases(chords, gaps, duration, sr)
    sitar = apply_reverb(sitar, sr, decay=0.3, delay_ms=45, mix=0.25)
    sitar = fade_in(sitar, sr, duration=6.0)  # sitar enters last

    # Apply dynamics to all instruments
    for inst_name, inst_audio in [('tabla', tabla), ('shaker', shaker),
                                   ('strings', strings), ('bansuri', bansuri),
                                   ('santoor', santoor), ('sitar', sitar)]:
        inst_audio *= dynamics

    # ── Stereo Mix ────────────────────────────────────────────────────────
    print("\n[7/7] Creating stereo mix...")

    # Instrument mix levels and panning (0=left, 1=right)
    mix_config = {
        'tabla':   {'level': 0.50, 'pan': 0.50},  # center
        'shaker':  {'level': 0.12, 'pan': 0.65},  # slight right
        'strings': {'level': 0.55, 'pan': 0.50},  # center (spread via internal voicing)
        'bansuri': {'level': 0.40, 'pan': 0.30},  # left
        'santoor': {'level': 0.30, 'pan': 0.70},  # right
        'sitar':   {'level': 0.28, 'pan': 0.60},  # slight right
    }

    inst_map = {
        'tabla': tabla, 'shaker': shaker, 'strings': strings,
        'bansuri': bansuri, 'santoor': santoor, 'sitar': sitar
    }

    n_samples = len(y)
    left = np.zeros(n_samples)
    right = np.zeros(n_samples)

    for name, audio in inst_map.items():
        cfg = mix_config[name]
        pan = cfg['pan']
        level = cfg['level']
        # Ensure same length
        a = audio[:n_samples] if len(audio) >= n_samples else np.pad(audio, (0, n_samples - len(audio)))
        a = a * level
        left += a * np.cos(pan * np.pi / 2)
        right += a * np.sin(pan * np.pi / 2)

    # Normalize orchestra
    orch_peak = max(np.abs(left).max(), np.abs(right).max())
    if orch_peak > 0:
        left = left / orch_peak * ORCHESTRA_LEVEL
        right = right / orch_peak * ORCHESTRA_LEVEL

    # Add vocals (centered)
    vocal = y * VOCAL_LEVEL
    left += vocal
    right += vocal

    # Create stereo output with slight string spread
    # Add a very subtle spread to strings: delay right channel by ~1ms
    spread_samples = int(0.001 * sr)
    right_spread = np.zeros_like(right)
    right_spread[spread_samples:] += strings[:n_samples-spread_samples] * 0.05
    left += strings[:n_samples] * 0.05 * -1  # cancel center, add sides
    right += right_spread

    stereo = np.column_stack([left, right])

    # Final limiting
    peak = np.abs(stereo).max()
    if peak > 0.95:
        stereo = stereo / peak * 0.95

    # Write WAV
    sf.write(output_path, stereo, sr)
    fsize = os.path.getsize(output_path)

    print(f"\n{'=' * 65}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Size: {fsize/1024:.0f} KB | Duration: {duration:.1f}s")
    print(f"  Stereo | {sr}Hz | 16-bit WAV")
    print(f"  Instruments: Tabla, Shaker, Strings, Bansuri, Santoor, Sitar")
    print(f"  Arrangement: Tasteful \u2014 instruments enter gradually")
    print(f"{'=' * 65}")

    return output_path


if __name__ == "__main__":
    main()
