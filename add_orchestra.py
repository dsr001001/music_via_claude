#!/usr/bin/env python3
"""
Orchestra Accompaniment Generator
Analyzes a vocal recording and adds classical string ensemble accompaniment.
Outputs a stereo file with vocals centered and strings spread across the stereo field.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
import os

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE = "Tumse_milkar_na_janekyu.m4a"
OUTPUT_FILE = "Tumse_milkar_na_janekyu_with_orchestra.wav"
SAMPLE_RATE = 48000

# Mix levels (0.0 to 1.0)
VOCAL_LEVEL = 1.0
ORCHESTRA_LEVEL = 0.38   # balanced but vocals remain the focus

# Reverb settings for strings
REVERB_DECAY = 0.3
REVERB_DELAY_MS = 40
REVERB_MIX = 0.25

# ── Chord Detection ──────────────────────────────────────────────────────────

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord templates: each chord is represented as a 12-element binary vector
CHORD_TEMPLATES = {}
for root in range(12):
    # Major triads
    major = np.zeros(12)
    major[root] = 1.0
    major[(root + 4) % 12] = 0.8
    major[(root + 7) % 12] = 0.8
    CHORD_TEMPLATES[f"{NOTE_NAMES[root]}"] = major

    # Minor triads
    minor = np.zeros(12)
    minor[root] = 1.0
    minor[(root + 3) % 12] = 0.8
    minor[(root + 7) % 12] = 0.8
    CHORD_TEMPLATES[f"{NOTE_NAMES[root]}m"] = minor

    # Seventh chords
    seventh = np.zeros(12)
    seventh[root] = 1.0
    seventh[(root + 4) % 12] = 0.7
    seventh[(root + 7) % 12] = 0.7
    seventh[(root + 10) % 12] = 0.6
    CHORD_TEMPLATES[f"{NOTE_NAMES[root]}7"] = seventh

    # Minor seventh chords
    m7 = np.zeros(12)
    m7[root] = 1.0
    m7[(root + 3) % 12] = 0.7
    m7[(root + 7) % 12] = 0.7
    m7[(root + 10) % 12] = 0.6
    CHORD_TEMPLATES[f"{NOTE_NAMES[root]}m7"] = m7


def detect_chords(y, sr, hop_length=512, segment_duration=0.5):
    """Detect chord progression from audio using chroma features."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Smooth chroma to reduce noise
    chroma_smooth = uniform_filter1d(chroma, size=10, axis=1)

    segment_frames = int(segment_duration * sr / hop_length)
    n_segments = chroma_smooth.shape[1] // segment_frames

    chords = []
    for seg in range(n_segments):
        start = seg * segment_frames
        end = start + segment_frames
        seg_chroma = chroma_smooth[:, start:end].mean(axis=1)

        # Normalize
        norm = np.linalg.norm(seg_chroma)
        if norm > 0:
            seg_chroma = seg_chroma / norm

        # Match against templates
        best_chord = "C"
        best_score = -1
        for name, template in CHORD_TEMPLATES.items():
            t_norm = template / np.linalg.norm(template)
            score = np.dot(seg_chroma, t_norm)
            if score > best_score:
                best_score = score
                best_chord = name

        time_start = seg * segment_duration
        chords.append((time_start, segment_duration, best_chord, best_score))

    return chords


def smooth_chord_progression(chords, min_duration=1.5):
    """Smooth rapid chord changes - hold chords for at least min_duration."""
    if not chords:
        return chords

    smoothed = [chords[0]]
    accumulated_duration = chords[0][1]

    for i in range(1, len(chords)):
        time, dur, chord, score = chords[i]
        prev_chord = smoothed[-1][2]

        if chord == prev_chord:
            # Same chord: extend duration
            old = smoothed[-1]
            smoothed[-1] = (old[0], old[1] + dur, old[2], max(old[3], score))
            accumulated_duration += dur
        elif accumulated_duration >= min_duration:
            # Different chord and previous held long enough: start new
            smoothed.append((time, dur, chord, score))
            accumulated_duration = dur
        else:
            # Extend previous chord (too short to change)
            old = smoothed[-1]
            smoothed[-1] = (old[0], old[1] + dur, old[2], old[3])
            accumulated_duration += dur

    return smoothed


# ── String Synthesis ──────────────────────────────────────────────────────────

def note_to_freq(note_name, octave):
    """Convert note name and octave to frequency in Hz."""
    semitone = NOTE_NAMES.index(note_name.replace('b', '').replace('#', ''))
    if '#' in note_name:
        semitone = (semitone) % 12  # already handled by NOTE_NAMES lookup
    midi = semitone + (octave + 1) * 12
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def chord_to_notes(chord_name):
    """Convert chord name to list of note semitones (relative to C)."""
    # Parse root and quality
    if chord_name.endswith('m7'):
        root_name = chord_name[:-2]
        intervals = [0, 3, 7, 10]
    elif chord_name.endswith('7'):
        root_name = chord_name[:-1]
        intervals = [0, 4, 7, 10]
    elif chord_name.endswith('m'):
        root_name = chord_name[:-1]
        intervals = [0, 3, 7]
    else:
        root_name = chord_name
        intervals = [0, 4, 7]

    root = NOTE_NAMES.index(root_name)
    return [(root + iv) % 12 for iv in intervals]


def synthesize_string_tone(freq, duration, sr, n_voices=3, vibrato_rate=5.5,
                           vibrato_depth=0.003, brightness=0.4,
                           attack=0.12, release=0.15):
    """
    Synthesize a single string-like tone using filtered sawtooth waves
    with vibrato and multiple detuned voices for ensemble richness.
    """
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    # ADSR envelope
    attack_samples = int(attack * sr)
    release_samples = int(release * sr)
    sustain_samples = n_samples - attack_samples - release_samples

    envelope = np.ones(n_samples)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples > 0 and sustain_samples > 0:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    elif sustain_samples <= 0:
        # Very short note - just fade in/out
        mid = n_samples // 2
        envelope[:mid] = np.linspace(0, 1, mid)
        envelope[mid:] = np.linspace(1, 0, n_samples - mid)

    # Vibrato LFO
    vibrato = vibrato_depth * freq * np.sin(2 * np.pi * vibrato_rate * t)

    # Generate multiple slightly detuned voices
    output = np.zeros(n_samples)
    detune_cents = [-8, 0, 8]  # slight detuning for richness
    if n_voices == 1:
        detune_cents = [0]
    elif n_voices == 2:
        detune_cents = [-6, 6]

    for detune in detune_cents[:n_voices]:
        # Detune frequency
        voice_freq = freq * (2.0 ** (detune / 1200.0))

        # Phase accumulator for sawtooth
        phase = np.cumsum(2 * np.pi * (voice_freq + vibrato) / sr)

        # Generate sawtooth and apply basic filtering
        saw = signal.sawtooth(phase)

        # Add some sine component for warmth
        sine = np.sin(phase)

        # Blend: more sine = warmer, more saw = brighter
        voice = (1 - brightness) * sine + brightness * saw
        output += voice / n_voices

    # Apply low-pass filter to soften the sound (simulate string body resonance)
    cutoff = min(freq * 6, sr * 0.45)  # cutoff relative to fundamental
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    if normalized_cutoff < 1.0:
        b, a = signal.butter(3, normalized_cutoff, btype='low')
        output = signal.filtfilt(b, a, output)

    # Apply envelope
    output *= envelope

    return output


def synthesize_string_chord(chord_name, octave_base, duration, sr,
                            instrument='violin', n_ensemble=2):
    """
    Synthesize a full string chord for a specific instrument range.

    instrument: 'cello', 'viola', 'violin1', 'violin2'
    """
    notes = chord_to_notes(chord_name)

    # Instrument-specific settings
    settings = {
        'cello': {
            'octave': octave_base,
            'brightness': 0.25,
            'vibrato_depth': 0.002,
            'vibrato_rate': 5.0,
            'attack': 0.2,
            'n_voices': 2,
            'play_notes': [0],  # just the root
        },
        'viola': {
            'octave': octave_base + 1,
            'brightness': 0.30,
            'vibrato_depth': 0.003,
            'vibrato_rate': 5.5,
            'attack': 0.15,
            'n_voices': 2,
            'play_notes': [0, 1] if len(notes) >= 2 else [0],  # root + third
        },
        'violin2': {
            'octave': octave_base + 1,
            'brightness': 0.35,
            'vibrato_depth': 0.003,
            'vibrato_rate': 5.8,
            'attack': 0.12,
            'n_voices': 3,
            'play_notes': [1, 2] if len(notes) >= 3 else [0, 1],  # third + fifth
        },
        'violin1': {
            'octave': octave_base + 2,
            'brightness': 0.35,
            'vibrato_depth': 0.004,
            'vibrato_rate': 6.0,
            'attack': 0.10,
            'n_voices': 3,
            'play_notes': [0, 2] if len(notes) >= 3 else [0],  # root + fifth (upper)
        },
    }

    s = settings.get(instrument, settings['violin1'])
    output = np.zeros(int(duration * sr))

    for idx in s['play_notes']:
        if idx < len(notes):
            semitone = notes[idx]
            freq = 440.0 * (2.0 ** ((semitone - 9 + (s['octave'] - 4) * 12) / 12.0))

            # Synthesize with slight random variation for naturalism
            for ens in range(n_ensemble):
                detune_offset = np.random.uniform(-3, 3)  # cents
                ens_freq = freq * (2.0 ** (detune_offset / 1200.0))
                tone = synthesize_string_tone(
                    ens_freq, duration, sr,
                    n_voices=s['n_voices'],
                    vibrato_rate=s['vibrato_rate'] + np.random.uniform(-0.3, 0.3),
                    vibrato_depth=s['vibrato_depth'],
                    brightness=s['brightness'],
                    attack=s['attack'],
                )
                output += tone / n_ensemble

    return output


# ── Reverb ────────────────────────────────────────────────────────────────────

def apply_reverb(audio, sr, decay=0.3, delay_ms=40, mix=0.25, n_taps=6):
    """Apply a simple multi-tap delay reverb to simulate room acoustics."""
    output = audio.copy()
    for tap in range(1, n_taps + 1):
        delay_samples = int(tap * delay_ms * sr / 1000)
        tap_gain = decay ** tap
        if delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * tap_gain
            output += delayed * mix
    return output


# ── Dynamic Envelope Follower ─────────────────────────────────────────────────

def compute_dynamics_envelope(y, sr, hop_length=512, smooth_window=2.0):
    """
    Compute a smoothed dynamics envelope from the vocal track.
    Used to make the orchestra follow the vocalist's dynamics.
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Smooth heavily
    smooth_frames = int(smooth_window * sr / hop_length)
    rms_smooth = uniform_filter1d(rms, size=max(smooth_frames, 1))

    # Normalize to 0-1 range
    max_rms = rms_smooth.max()
    if max_rms > 0:
        rms_smooth = rms_smooth / max_rms

    # Interpolate to sample level
    times = librosa.frames_to_time(np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length)
    t_samples = np.arange(len(y)) / sr
    envelope = np.interp(t_samples, times, rms_smooth)

    # Apply gentle compression: boost quiet parts, tame loud parts
    envelope = np.clip(envelope, 0.15, 1.0)
    envelope = 0.3 + 0.7 * envelope  # minimum level of 0.3

    return envelope


# ── Crossfade Between Chords ──────────────────────────────────────────────────

def crossfade_chords(prev_audio, next_audio, crossfade_samples):
    """Crossfade between two chord segments to avoid clicks."""
    if crossfade_samples <= 0 or len(prev_audio) < crossfade_samples:
        return prev_audio

    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)

    # Apply fade out to end of previous
    result = prev_audio.copy()
    result[-crossfade_samples:] *= fade_out

    # Apply fade in to start of next and add
    if len(next_audio) >= crossfade_samples:
        blend = next_audio[:crossfade_samples] * fade_in
        result[-crossfade_samples:] += blend

    return result


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    print("=" * 60)
    print("  Orchestra Accompaniment Generator")
    print("  Classical String Ensemble Edition")
    print("=" * 60)

    # ── Step 1: Load audio ────────────────────────────────────────────────
    print("\n[1/6] Loading audio...")
    y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"  Duration: {duration:.2f}s | Sample rate: {sr}Hz | Samples: {len(y)}")

    # ── Step 2: Detect chords ─────────────────────────────────────────────
    print("\n[2/6] Analyzing harmony and detecting chords...")
    raw_chords = detect_chords(y, sr, segment_duration=0.5)
    chords = smooth_chord_progression(raw_chords, min_duration=1.5)

    print(f"  Detected {len(chords)} chord regions:")
    for time_start, dur, chord, score in chords:
        print(f"    {time_start:6.1f}s - {time_start+dur:6.1f}s: {chord:4s} (confidence: {score:.2f})")

    # ── Step 3: Compute dynamics envelope ─────────────────────────────────
    print("\n[3/6] Computing dynamics envelope...")
    dynamics = compute_dynamics_envelope(y, sr)
    print(f"  Envelope range: {dynamics.min():.2f} - {dynamics.max():.2f}")

    # ── Step 4: Synthesize string ensemble ────────────────────────────────
    print("\n[4/6] Synthesizing string ensemble...")
    instruments = ['cello', 'viola', 'violin2', 'violin1']
    instrument_audio = {inst: np.zeros(len(y)) for inst in instruments}

    crossfade_duration = 0.08  # 80ms crossfade between chords
    crossfade_samples = int(crossfade_duration * sr)

    for chord_idx, (time_start, dur, chord, score) in enumerate(chords):
        start_sample = int(time_start * sr)
        end_sample = min(int((time_start + dur) * sr), len(y))
        seg_duration = (end_sample - start_sample) / sr

        print(f"  Chord {chord_idx+1}/{len(chords)}: {chord} ({time_start:.1f}s-{time_start+dur:.1f}s)")

        for inst in instruments:
            # Determine base octave based on estimated key
            if inst == 'cello':
                base_oct = 2
            elif inst == 'viola':
                base_oct = 3
            else:
                base_oct = 3  # violin base, shifted up inside the function

            tone = synthesize_string_chord(
                chord, base_oct, seg_duration, sr,
                instrument=inst, n_ensemble=2
            )

            # Ensure tone length matches segment
            if len(tone) > end_sample - start_sample:
                tone = tone[:end_sample - start_sample]
            elif len(tone) < end_sample - start_sample:
                tone = np.pad(tone, (0, end_sample - start_sample - len(tone)))

            # Crossfade with previous chord
            if chord_idx > 0 and start_sample > crossfade_samples:
                fade_region = instrument_audio[inst][start_sample - crossfade_samples:start_sample]
                if len(fade_region) == crossfade_samples and len(tone) >= crossfade_samples:
                    fade_out = np.linspace(1, 0, crossfade_samples)
                    fade_in = np.linspace(0, 1, crossfade_samples)
                    instrument_audio[inst][start_sample - crossfade_samples:start_sample] *= fade_out
                    tone[:crossfade_samples] *= fade_in

            instrument_audio[inst][start_sample:start_sample + len(tone)] += tone

    # ── Step 5: Apply effects and dynamics ────────────────────────────────
    print("\n[5/6] Applying dynamics, reverb, and mixing...")

    # Apply dynamics envelope to each instrument
    for inst in instruments:
        instrument_audio[inst] *= dynamics

    # Apply reverb to each instrument
    for inst in instruments:
        instrument_audio[inst] = apply_reverb(
            instrument_audio[inst], sr,
            decay=REVERB_DECAY, delay_ms=REVERB_DELAY_MS, mix=REVERB_MIX
        )

    # Normalize each instrument
    for inst in instruments:
        peak = np.abs(instrument_audio[inst]).max()
        if peak > 0:
            instrument_audio[inst] = instrument_audio[inst] / peak * 0.7

    # ── Step 6: Stereo mixing ─────────────────────────────────────────────
    print("\n[6/6] Creating stereo mix...")

    # Panning positions (0 = hard left, 1 = hard right)
    panning = {
        'cello':   0.50,  # center (bass should be centered)
        'viola':   0.35,  # slightly left
        'violin2': 0.25,  # left
        'violin1': 0.75,  # right
    }

    # Instrument relative levels
    inst_levels = {
        'cello':   0.55,
        'viola':   0.50,
        'violin2': 0.45,
        'violin1': 0.45,
    }

    # Create stereo orchestra
    left = np.zeros(len(y))
    right = np.zeros(len(y))

    for inst in instruments:
        pan = panning[inst]
        level = inst_levels[inst]
        audio = instrument_audio[inst] * level

        # Equal-power panning
        left += audio * np.cos(pan * np.pi / 2)
        right += audio * np.sin(pan * np.pi / 2)

    # Scale orchestra
    orchestra_peak = max(np.abs(left).max(), np.abs(right).max())
    if orchestra_peak > 0:
        left = left / orchestra_peak * ORCHESTRA_LEVEL
        right = right / orchestra_peak * ORCHESTRA_LEVEL

    # Add vocals (centered)
    vocal = y * VOCAL_LEVEL
    left += vocal
    right += vocal

    # Final stereo array
    stereo = np.column_stack([left, right])

    # Final normalization (prevent clipping)
    peak = np.abs(stereo).max()
    if peak > 0.95:
        stereo = stereo / peak * 0.95

    # Write output
    sf.write(output_path, stereo, sr)
    output_size = os.path.getsize(output_path)

    print(f"\n{'=' * 60}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Size: {output_size / 1024:.0f} KB")
    print(f"  Duration: {duration:.1f}s | Stereo | {sr}Hz | 16-bit WAV")
    print(f"  Vocals: center | Cello: center | Viola: slight L")
    print(f"  Violin 2: left | Violin 1: right")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
