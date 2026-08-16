"""Windowing and the 149-number feature vector."""

import hashlib

import librosa
import numpy as np
from scipy.signal import sosfiltfilt

from .config import CACHE_DIR, HOP, MEL_HZ, N_FEATURES, N_MELS, N_FFT, SOS, SR, STEP, WIN


def windows(y):
    """4 s windows, 2 s hop.

    Short clips are tiled, not zero-padded: silence would fake the quiet gaps
    between beats and read as a healthy heart.
    """
    if len(y) < SR // 2:  # under half a second is not a heartbeat, it is a dropout
        raise ValueError(f"clip too short to screen: {len(y)} samples")
    if len(y) < WIN:
        y = np.tile(y, int(np.ceil(WIN / len(y))))
    return [y[i:i + WIN] for i in range(0, len(y) - WIN + 1, STEP)]


def features(w):
    """149 numbers for one 4 s window."""
    w = sosfiltfilt(SOS, w)
    w = np.ascontiguousarray(w / (np.abs(w).max() + 1e-9), dtype=np.float32)

    mel = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, fmax=1000)
    logmel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=13)
    dmfcc = librosa.feature.delta(mfcc)

    spec = np.abs(librosa.stft(w, n_fft=N_FFT, hop_length=HOP))
    shape = np.vstack([
        librosa.feature.spectral_centroid(S=spec, sr=SR),
        librosa.feature.spectral_bandwidth(S=spec, sr=SR),
        librosa.feature.spectral_rolloff(S=spec, sr=SR),
        librosa.feature.spectral_flatness(S=spec),
        librosa.feature.rms(S=spec, frame_length=N_FFT),
        librosa.feature.zero_crossing_rate(w, frame_length=N_FFT, hop_length=HOP),
    ])

    # Murmur cues. A healthy heart is two brief low-frequency thumps (S1/S2) with
    # little in between. A systolic murmur -- what aortic stenosis makes -- fills
    # the interval after S1 with high-frequency turbulence, so frames that are
    # neither peak nor silence get brighter and the beat gets longer. Directions
    # below were MEASURED on labelled audio, not assumed; the self-check
    # re-verifies them.
    env = logmel.mean(0)
    lo, hi = logmel[MEL_HZ < 150].mean(0), logmel[MEL_HZ >= 150].mean(0)
    p10, p40, p85, p90, p95 = np.percentile(env, [10, 40, 85, 90, 95])
    mid = (env >= p40) & (env <= p85)
    cue = [
        (hi - lo).mean(),                               # brightness       murmur higher
        (env > env.max() - 10).mean(),                  # duty cycle       murmur higher
        (hi[mid] - lo[mid]).mean(),                     # non-peak bright  murmur higher
        p95 - p10,                                      # dynamic range    murmur higher
        hi[env <= p40].mean() - hi[env >= p90].mean(),  # HF gaps-vs-peaks murmur LOWER
    ]

    return np.concatenate([logmel.mean(1), logmel.std(1), mfcc.mean(1), mfcc.std(1),
                           dmfcc.mean(1), dmfcc.std(1), shape.mean(1), shape.std(1), cue])


def build_matrix(df):
    """Window features plus the recording each window came from. Cached on disk."""
    key = hashlib.sha1(("".join(df.fname) + str(N_FEATURES)).encode()).hexdigest()
    cache = CACHE_DIR / f"feat-{key[:16]}.npz"
    if cache.exists():
        c = np.load(cache)
        return c["X"], c["owner"]

    rows, owner = [], []
    for i, path in enumerate(df.path):
        ws = windows(load_audio(path))
        rows.extend(features(w) for w in ws)
        owner.extend([i] * len(ws))
        if (i + 1) % 250 == 0:
            print(f"  featurised {i + 1}/{len(df)} recordings", flush=True)

    X, owner = np.vstack(rows), np.array(owner)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, owner=owner)
    return X, owner
