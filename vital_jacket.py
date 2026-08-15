#!/usr/bin/env python
"""Vital Jacket -- heart-sound murmur screener, end to end in one file.

    python vital_jacket.py            # fetch data, train, evaluate, write every report
    python vital_jacket.py --help     # stage selection and options

Produces, in reports/:
    cohort.png          validation charts: ROC, score separation, confusion, calibration
    patient_*.png       per-probe review pages, one per clinical outcome
    detections.csv      every recording ranked by score, with its verdict
    review.html         playable clinical review page (needs ffmpeg)

WHAT THIS DETECTS
    Valve blockage (stenosis). A narrowed or leaking valve forces blood through a
    small opening; the turbulence is audible as a murmur, and that is what the
    model is trained on.

WHAT IT DOES NOT DETECT
    Coronary artery blockage -- the kind that causes heart attacks. Training that
    needs recordings labelled against coronary angiography, which no open dataset
    has. Nor does it detect "heart block" in the cardiology sense (AV conduction
    block), which is an ECG finding, not a sound.

    A murmur is a finding, not a diagnosis. The honest output is "this chest
    sounds abnormal, send it for an echocardiogram".

Data: CirCor DigiScope Phonocardiogram Dataset v1.0.3 (Oliveira et al., PhysioNet),
de-identified, ODC-By 1.0 -- redistributable with attribution. Optional: the PASCAL
Classifying Heart Sounds Challenge set, if present under Heart Sound/dataset/.

Not a medical device. Not validated, not certified, not for clinical use.
"""

import argparse
import base64
import hashlib
import io
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import joblib
import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
HEART = ROOT_DIR / "Heart Sound"
PASCAL = HEART / "dataset"
CIRCOR = HEART / "circor"
CINC = HEART / "cinc2016"
CACHE_DIR = HEART / ".cache"
OUT = ROOT_DIR / "reports"
MODEL = ROOT_DIR / "murmur_model.joblib"

SR = 4000              # CirCor and PASCAL set_b are native 4 kHz; set_a resamples down
N_MELS, N_FFT, HOP = 40, 512, 128
WIN, STEP = 4 * SR, 2 * SR   # 4 s analysis windows, 50% overlap
N_FEATURES = 149
CLIP_SECONDS = 14      # audio excerpt on the review page

CIRCOR_URL = "https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/"
CIRCOR_TOP = "the-circor-digiscope-phonocardiogram-dataset-1.0.3/"
CINC_URL = "https://physionet.org/files/challenge-2016/1.0.0/training.zip"

# 25-900 Hz for analysis: below is body/handling rumble, above is past anything
# diagnostic at this rate. Murmurs live around 100-600 Hz.
SOS = butter(4, (25, 900), btype="bandpass", fs=SR, output="sos")
# Wider for playback -- the band an electronic stethoscope passes.
PLAYBACK_SOS = butter(4, (20, 1000), btype="bandpass", fs=SR, output="sos")
MEL_HZ = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=1000)

# Validated light-mode data-viz palette. The spectrogram ramp is deliberately
# SINGLE HUE: a viridis/jet rainbow on a magnitude scale creates banding that
# reads as structure in the signal, which on a diagnostic image is a correctness
# problem, not a cosmetic one.
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, CRITICAL, GOOD = "#2a78d6", "#eb6834", "#d03b3b", "#0ca30c"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("seq_blue", SEQ)
DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", CRITICAL])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",  # never dashed
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 130,
})

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

# PASCAL set_b encodes <label>_<patient>_<epoch_ms>_<position>.wav. Its 149-file
# noisy subset separates the label with ONE underscore (normal_noisynormal_101_...)
# while everything else uses two -- so match the patient/epoch/position tail only.
_SET_B = re.compile(r"_(\d{3})_\d{10,}_([A-Z])")


def patient_and_position(fname):
    """Patient id and auscultation position. PASCAL set_a has neither, so each
    clip becomes its own patient group -- conservative, keeps grouping honest."""
    m = _SET_B.search(fname)
    if m:
        return f"b{m.group(1)}", m.group(2)
    return f"a{Path(fname).stem}", "?"


def fetch_circor():
    """Download CirCor (449 MB) if absent. Idempotent."""
    if (CIRCOR / "training_data.csv").exists():
        n = len(list((CIRCOR / "training_data").glob("*.wav")))
        print(f"CirCor already present: {n} recordings")
        return
    CIRCOR.mkdir(parents=True, exist_ok=True)
    zpath = CIRCOR / "circor.zip"
    print("Downloading CirCor DigiScope, 449 MB from physionet.org ...")
    with urllib.request.urlopen(CIRCOR_URL) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    print("Extracting ...")
    keep_exact = {"training_data.csv", "LICENSE.txt", "RECORDS"}
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.startswith(CIRCOR_TOP) or name.endswith("/"):
                continue
            rel = name[len(CIRCOR_TOP):]
            if not (rel.endswith(".wav") or rel in keep_exact):
                continue
            out = CIRCOR / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zpath.unlink()
    print(f"Done: {len(list((CIRCOR / 'training_data').glob('*.wav')))} recordings")


def fetch_cinc():
    """Download the PhysioNet/CinC 2016 training set (181 MB) if absent."""
    # Count the recordings, do not just look for a REFERENCE.csv. A partial
    # extraction leaves the csv in place, and a presence check would then skip
    # the download and quietly train on a fraction of the corpus.
    have = len(list(CINC.glob("training-*/*.wav")))
    want = sum(len([l for l in ref.read_text().splitlines() if l.strip()])
               for ref in CINC.glob("training-*/REFERENCE.csv"))
    if have and have >= want and len(list(CINC.glob("training-*/REFERENCE.csv"))) >= 6:
        print(f"CinC 2016 already present: {have} recordings")
        return
    if have:
        print(f"CinC 2016 incomplete ({have} recordings) -- refetching")
    CINC.mkdir(parents=True, exist_ok=True)
    zpath = CINC / "training.zip"
    print("Downloading PhysioNet/CinC 2016 training set, 181 MB ...")
    with urllib.request.urlopen(CINC_URL) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    print("Extracting ...")
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.endswith("/") or not (name.endswith(".wav")
                                          or name.endswith("REFERENCE.csv")):
                continue
            out = CINC / Path(name).parent.name / Path(name).name
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zpath.unlink()
    print(f"Done: {len(list(CINC.glob('training-*/*.wav')))} recordings")


def _cinc_index(labels):
    """PhysioNet/CinC 2016: 3126 recordings, five sub-databases, ADULTS INCLUDED.

    Held out as a generalisation test, never mixed into training, because its
    labels answer a DIFFERENT question. CirCor labels "is a murmur audible at
    this site"; CinC labels "does this patient have a confirmed cardiac
    diagnosis", and its abnormal group explicitly includes coronary artery
    disease -- which produces no murmur and which this model cannot hear.

    So CinC "abnormal" is a superset of what the model detects. Transfer scores
    against it are a FLOOR, not a like-for-like comparison: the model is
    penalised for missing patients it was never built to find. Read a decent
    score as encouraging and a poor one as ambiguous.

    Patient ids are not published, so each recording forms its own group. That
    is fine for a held-out test set -- nothing is being split.
    """
    rows = []
    for ref in sorted(CINC.glob("training-*/REFERENCE.csv")):
        sub = ref.parent.name
        for line in ref.read_text().splitlines():
            if not line.strip():
                continue
            name, val = line.split(",")[:2]
            wav = ref.parent / f"{name}.wav"
            if not wav.exists():
                continue
            # -1 normal, 1 abnormal.
            label = "murmur" if val.strip() == "1" else "normal"
            if label in labels:
                rows.append({"fname": f"cinc2016/{sub}/{name}.wav", "label": label,
                             "patient": f"n{sub[-1]}{name}", "position": "?",
                             "path": wav, "dataset": f"cinc_{sub[-1]}"})
    if not rows:
        raise FileNotFoundError(
            f"CinC 2016 not found at {CINC}. Fetch it with --fetch-cinc.")
    return pd.DataFrame(rows)


def _pascal_index(which, labels):
    df = pd.read_csv(PASCAL / f"set_{which}.csv")
    df = df[df.label.isin(labels)].copy()
    df["patient"], df["position"] = zip(*df.fname.map(patient_and_position))
    df["path"] = df.fname.map(lambda f: PASCAL / f)
    df["dataset"] = f"pascal_{which}"
    return df


def _circor_index(labels):
    """CirCor: 942 patients, ~3163 recordings at sites AV/PV/TV/MV.

    Murmur is graded per patient, but `Murmur locations` says which probes
    actually heard it. Patient 13918 records at AV+PV+TV+MV with the murmur
    audible only at TV -- labelling all four "murmur" would teach the model that
    three clean recordings are murmurs. Each recording is labelled by whether ITS
    site heard anything, which is what a per-probe detector needs; patient-level
    fusion then recovers the patient via the one probe that did.

    Murmur=Unknown (68 patients, 156 recordings) is dropped: the annotators could
    not tell, so there is no label to learn from.
    """
    csv = CIRCOR / "training_data.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"CirCor not found at {CIRCOR}. Run `python vital_jacket.py` without "
            "--skip-fetch, or use --sets a,b for the bundled PASCAL data.")
    rows = []
    for r in pd.read_csv(csv).to_dict("records"):
        if r["Murmur"] == "Unknown":
            continue
        heard = set(str(r["Murmur locations"]).split("+"))
        for pos in str(r["Recording locations:"]).split("+"):
            wav = CIRCOR / "training_data" / f"{r['Patient ID']}_{pos}.wav"
            if not wav.exists():
                continue
            label = "murmur" if (r["Murmur"] == "Present" and pos in heard) else "normal"
            if label in labels:
                rows.append({"fname": f"circor/{wav.name}", "label": label,
                             "patient": f"c{r['Patient ID']}", "position": pos,
                             "path": wav, "dataset": "circor"})
    return pd.DataFrame(rows)


def load_index(sets=("circor",), labels=("normal", "murmur")):
    """Index of recordings: path, label, patient, position, source set.

    Callers MUST group by `patient`. CirCor gives each patient ~4 probes and
    PASCAL set_b has 165 patients across 461 recordings; split those at random
    and the same patient lands in train and test, inflating every score.
    """
    frames = [_circor_index(labels) if s == "circor"
              else _cinc_index(labels) if s == "cinc"
              else _pascal_index(s, labels) for s in sets]
    df = pd.concat(frames, ignore_index=True)
    missing = [p for p in df.path if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} indexed files missing, e.g. {missing[0]}")
    return df.reset_index(drop=True)


def load_audio(path):
    """Mono waveform at SR. librosa resamples PASCAL set_a's 44.1 kHz down for us,
    so features from the two sources are comparable at all."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


# ---------------------------------------------------------------------------
# 2. Audio analytics -- what is actually in the recording
# ---------------------------------------------------------------------------
#
# The classifier answers "murmur or not" and nothing else. This section answers
# the questions a person asks next: how fast is the heart, where are the beats,
# is the recording even usable, and -- the one that matters clinically -- does
# the extra sound fall in systole or diastole. That single distinction separates
# aortic stenosis from aortic regurgitation, so a screener that cannot say which
# is withholding the most useful thing it knows.

def shannon_envelope(y, smooth_ms=40):
    """Shannon energy envelope -- the standard PCG beat-detection front end.

    -x^2 log(x^2) weights medium-amplitude sound (heart sounds) above both the
    quiet noise floor and the occasional loud click, which plain rectification
    or x^2 would let dominate.
    """
    # Normalise on the 99.5th percentile, not the peak. One handling click can
    # be 8x the rest of the recording; dividing by it squashes every real heart
    # sound into the bottom of the range and the envelope goes flat, which then
    # reads as "no beats" downstream.
    x = np.clip(y / (np.percentile(np.abs(y), 99.5) + 1e-9), -1, 1)
    e = -(x ** 2) * np.log(x ** 2 + 1e-10)
    n = max(3, int(smooth_ms * SR / 1000) | 1)
    e = np.convolve(e, np.hanning(n) / np.hanning(n).sum(), mode="same")
    return (e - e.min()) / (np.ptp(e) + 1e-9)


def detect_sounds(y, env=None, prominence=0.15, refractory=0.15):
    """Times of candidate heart sounds, in seconds.

    Selects on PROMINENCE, not height: a heart sound is a peak that stands out
    from its immediate surroundings, and an absolute threshold instead just
    tracks recording gain. Measured against PASCAL's hand-marked S1/S2 (390
    sounds, 21 files) an amplitude threshold plateaued at 62% precision;
    prominence lifts it to 86% with no loss of timing accuracy.

    Refractory 150 ms: no two distinct heart sounds occur closer than that even
    at 200 bpm, so anything nearer is one sound detected twice.

    The defaults sit mid-plateau -- F1 moves only 0.885 to 0.892 across a
    threefold range of prominence, so they are not tuned to those 21 files.
    validate_segmentation() re-measures this.
    """
    env = shannon_envelope(y) if env is None else env
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(env, prominence=prominence,
                          distance=int(refractory * SR))
    return peaks / SR, env


def heart_rate(sound_times):
    """Beats per minute from the S1-to-S1 interval.

    Sounds alternate S1, S2, S1, ... so one cardiac cycle spans TWO detections.
    Median rather than mean: a single missed or doubled beat would drag a mean
    badly, and these recordings routinely have one.
    """
    if len(sound_times) < 4:
        return None, None
    gaps = np.diff(sound_times)
    cycle = np.median(gaps) * 2
    return 60.0 / cycle, cycle


def segment_cycle(sound_times):
    """Label the detections S1/S2 and return the systolic and diastolic spans.

    The whole assignment rests on one physiological asymmetry: at rest systole
    (S1->S2) is shorter than diastole (S2->S1). So of the two possible
    alternating labellings, the correct one is whichever puts the SHORTER
    intervals between S1 and S2. This inverts at very high heart rates, where
    diastole shortens faster than systole -- flagged rather than silently wrong.
    """
    if len(sound_times) < 4:
        return [], [], []
    gaps = np.diff(sound_times)
    # Two interleaved phases; whichever has the smaller median is systole.
    even, odd = gaps[0::2], gaps[1::2]
    if len(even) == 0 or len(odd) == 0:
        return [], [], []
    start = 0 if np.median(even) <= np.median(odd) else 1
    labels = ["?"] * len(sound_times)
    for i in range(start, len(sound_times) - 1, 2):
        labels[i], labels[i + 1] = "S1", "S2"
    systole = [(sound_times[i], sound_times[i + 1])
               for i in range(start, len(sound_times) - 1, 2)]
    diastole = [(sound_times[i + 1], sound_times[i + 2])
                for i in range(start, len(sound_times) - 2, 2)]
    return labels, systole, diastole


def cycles_are_plausible(systole, diastole, bpm):
    """Does the segmentation describe a heart, or did detection fall apart?

    Guards the timing call. Without this, a recording whose beats were never
    found still produces confident systolic/diastolic spans -- and a sentence
    naming candidate valve lesions off a 10-second "diastole" is worse than
    saying nothing, because it reads as a clinical finding.
    """
    if not systole or not diastole or bpm is None:
        return False, "too few complete cycles"
    if not 40 <= bpm <= 200:
        return False, f"heart rate {bpm:.0f} bpm is outside a plausible range"
    s = float(np.median([b - a for a, b in systole]))
    d = float(np.median([b - a for a, b in diastole]))
    if not 0.15 <= s <= 0.55:
        return False, f"systole {s:.2f} s is outside 0.15-0.55 s"
    if not 0.15 <= d <= 1.50:
        return False, f"diastole {d:.2f} s is outside 0.15-1.50 s"
    # Beat-to-beat consistency: a real rhythm is regular enough that the spread
    # of its intervals stays well under their length. Scattered detections are not.
    spread = float(np.std([b - a for a, b in diastole]))
    if spread > 0.5 * d:
        return False, "cycle lengths are too irregular to time reliably"
    return True, ""


def murmur_timing(y, systole, diastole, env=None):
    """Where the extra sound sits: systole, diastole, or neither.

    Measures high-frequency energy (>150 Hz, above the S1/S2 thumps) in the
    middle of each interval, skipping 25% at each end so the heart sounds
    themselves are excluded. What remains between beats is turbulence, or
    nothing at all in a clean recording.
    """
    if not systole or not diastole:
        return None
    hf = sosfiltfilt(butter(4, (150, 900), btype="bandpass", fs=SR, output="sos"), y)
    hf = hf / (np.abs(hf).max() + 1e-9)

    def band_energy(spans):
        vals = []
        for a, b in spans:
            if b - a < 0.06:
                continue
            pad = (b - a) * 0.25
            i, j = int((a + pad) * SR), int((b - pad) * SR)
            if j > i:
                vals.append(float(np.sqrt(np.mean(hf[i:j] ** 2))))
        return float(np.median(vals)) if vals else 0.0

    sys_e, dia_e = band_energy(systole), band_energy(diastole)
    # Reference: energy inside the heart sounds themselves, so the ratio is
    # "how loud is the gap compared to the beat" rather than an absolute level
    # that would just track recording gain.
    beat_e = band_energy([(a, a + 0.08) for a, _ in systole]) + 1e-9
    return {"systolic": sys_e / beat_e, "diastolic": dia_e / beat_e,
            "systolic_raw": sys_e, "diastolic_raw": dia_e}


def signal_quality(y):
    """Is this recording worth scoring at all?

    The screener's worst failures are recordings where the heart sounds are
    buried, not where the model misjudged them -- so quality is reported
    alongside the verdict rather than left for someone to infer.
    """
    clipped = float(np.mean(np.abs(y) >= 0.999 * np.abs(y).max()))
    x = np.clip(y / (np.percentile(np.abs(y), 99.5) + 1e-9), -1, 1)
    # Broadband clicks: samples far above the local level, the signature of
    # handling noise and probe friction.
    d = np.abs(np.diff(x))
    clicks = float(np.mean(d > 10 * np.median(d) + 1e-9))
    env = shannon_envelope(y)
    # Beat-to-gap contrast. A clean recording has loud beats and quiet gaps; a
    # noisy one flattens toward a constant level.
    contrast = float(np.percentile(env, 90) - np.percentile(env, 20))
    silent = float(np.mean(np.abs(x) < 0.005))
    verdict = ("good" if contrast > 0.35 and clicks < 0.02 else
               "usable" if contrast > 0.20 else "poor")
    return {"verdict": verdict, "contrast": contrast, "clicks": clicks,
            "clipped": clipped, "silent": silent}


def analyse(path, model=None):
    """Full step-by-step analysis of one recording, printed as it goes."""
    path = Path(path)
    raw = load_audio(path)
    y = sosfiltfilt(SOS, raw)
    y = np.ascontiguousarray(y / (np.abs(y).max() + 1e-9), dtype=np.float32)
    dur = len(y) / SR

    print(f"\n{'=' * 66}\n{path.name}\n{'=' * 66}")
    print(f"\n[1] Recording")
    print(f"    duration      {dur:.1f} s")
    print(f"    sample rate   {SR} Hz (resampled)")
    print(f"    bandpassed    25-900 Hz")

    q = signal_quality(raw)
    print(f"\n[2] Signal quality: {q['verdict'].upper()}")
    print(f"    beat-to-gap contrast  {q['contrast']:.2f}   (>0.35 good, <0.20 poor)")
    print(f"    click artifacts       {q['clicks']:.1%}      (handling noise)")
    print(f"    clipped samples       {q['clipped']:.1%}")
    if q["verdict"] == "poor":
        print("    -> heart sounds are buried; re-record before trusting any verdict")

    times, env = detect_sounds(y)
    bpm, cycle = heart_rate(times)
    print(f"\n[3] Heart sounds detected: {len(times)}")
    if bpm:
        print(f"    heart rate    {bpm:.0f} bpm   (cycle {cycle:.2f} s)")
        if bpm < 50 or bpm > 180:
            print("    -> outside 50-180 bpm; detection is probably miscounting")
    else:
        print("    too few sounds to estimate a rate")

    labels, systole, diastole = segment_cycle(times)
    print(f"\n[4] Cycle segmentation: {len(systole)} complete cycles")
    if systole and diastole:
        s = np.median([b - a for a, b in systole])
        d = np.median([b - a for a, b in diastole])
        print(f"    systole  (S1->S2)  {s:.3f} s")
        print(f"    diastole (S2->S1)  {d:.3f} s")
        print(f"    ratio              {s / d:.2f}   (below 1.0 is normal at rest)")
        if s > d:
            print("    -> systole longer than diastole: either a fast heart rate or"
                  "\n       a mislabelled cycle.")
        print(f"    first few:  " + "  ".join(
            f"{lab}@{t:.2f}s" for t, lab in list(zip(times, labels))[:6]))
    else:
        print("    no complete cycles found")

    ok, why = cycles_are_plausible(systole, diastole, bpm)
    mt = murmur_timing(y, systole, diastole) if ok else None
    print(f"\n[5] Where the extra sound falls")
    if not ok:
        print(f"    NOT REPORTED -- {why}.")
        print("    Timing needs reliable beat detection, and this recording did not"
              "\n    give it. The model verdict below does not depend on segmentation.")
    elif mt is None:
        print("    not enough clean cycles to place it")
    else:
        margin = mt["systolic"] - mt["diastolic"]
        print(f"    systolic gap energy   {mt['systolic']:.2f}  (relative to the beat)")
        print(f"    diastolic gap energy  {mt['diastolic']:.2f}")
        print(f"    systolic skew         {margin:+.2f}")
        # Calibrated on 108 labelled CirCor recordings: healthy hearts already
        # skew systolic by a median of +0.14, murmurs by +0.24. So this measure
        # answers WHICH PHASE, never WHETHER -- the absolute level does not
        # separate the two classes at all (normal median 1.13, murmur 0.99).
        # Anything reported here is conditional on the verdict in [7].
        if margin > 0.20:
            print("    -> SYSTOLIC-dominant. IF a murmur is present, this timing fits"
                  "\n       aortic/pulmonary stenosis or mitral/tricuspid regurgitation.")
        elif margin < -0.10:
            print("    -> DIASTOLIC-dominant. IF a murmur is present, this fits"
                  "\n       aortic/pulmonary regurgitation or mitral stenosis."
                  "\n       Diastolic murmurs are rarely innocent.")
        else:
            print("    -> No clear phase preference (healthy hearts skew +0.14 here"
                  "\n       by default, so this is within the normal range).")
        print("       Reference: healthy median +0.14, murmur median +0.24."
              "\n       Timing narrows a differential; it cannot establish one.")

    ws = windows(y)
    F = np.vstack([features(w) for w in ws])
    cues = F[:, -5:].mean(0)
    print(f"\n[6] Acoustic cues, averaged over {len(ws)} windows")
    for name, val, direction in zip(
            ["brightness", "duty cycle", "non-peak brightness",
             "dynamic range", "HF gaps vs peaks"], cues,
            ["higher = murmur", "higher = murmur", "higher = murmur",
             "higher = murmur", "lower = murmur"]):
        print(f"    {name:<22}{val:>8.2f}   ({direction})")

    print(f"\n[7] Model verdict")
    if model is None and MODEL.exists():
        model = joblib.load(MODEL)
    if model is None:
        print(f"    no model at {MODEL.name} -- run --train first")
    else:
        scores = model["model"].predict_proba(F)[:, 1]
        score, thr = float(scores.mean()), model["threshold"]
        print(f"    score        {score:.3f}   (threshold {thr:.3f})")
        print(f"    per window   " + " ".join(f"{s:.2f}" for s in scores[:12]))
        print(f"    VERDICT      {'MURMUR DETECTED' if score > thr else 'clear'}")
        if q["verdict"] == "poor":
            print("    Poor signal quality -- this verdict is unreliable.")
        print("\n    A flag means: refer for an echocardiogram. Not a diagnosis, and\n"
              "    it says nothing about coronary artery disease.")
    return {"duration": dur, "quality": q, "bpm": bpm, "cycles": len(systole),
            "timing": mt}


def validate_segmentation(tolerance=0.06):
    """Score the S1/S2 detector against PASCAL's hand-annotated ground truth.

    21 files, 195 S1 and 195 S2 marks. Without this the segmentation is just an
    assertion; the timing analysis it feeds would be unfalsifiable.
    """
    timing = PASCAL / "set_a_timing.csv"
    if not timing.exists():
        print("  (skipped -- set_a_timing.csv not present)")
        return None
    truth = pd.read_csv(timing)
    truth["t"] = truth.location / 44100.0      # location is a SAMPLE INDEX at 44.1 kHz

    hits = misses = extra = 0
    errors = []
    for fname, g in truth.groupby("fname"):
        wav = PASCAL / fname
        if not wav.exists():
            continue
        y = sosfiltfilt(SOS, load_audio(wav))
        y = np.ascontiguousarray(y / (np.abs(y).max() + 1e-9), dtype=np.float32)
        det, _ = detect_sounds(y)
        ref = np.sort(g["t"].to_numpy())
        used = np.zeros(len(det), bool)
        for t in ref:
            if len(det) == 0:
                misses += 1
                continue
            d = np.abs(det - t)
            i = int(np.argmin(d))
            if d[i] <= tolerance and not used[i]:
                hits += 1
                used[i] = True
                errors.append(det[i] - t)
            else:
                misses += 1
        extra += int((~used).sum())

    recall = hits / max(hits + misses, 1)
    precision = hits / max(hits + extra, 1)
    print(f"  S1/S2 detection vs {len(truth)} hand-marked sounds "
          f"(+/-{tolerance * 1000:.0f} ms):")
    print(f"    recall {recall:.1%} | precision {precision:.1%} | "
          f"median timing error {np.median(np.abs(errors)) * 1000:.0f} ms"
          if errors else "    no matches")
    return {"recall": recall, "precision": precision, "hits": hits,
            "misses": misses, "extra": extra}


# ---------------------------------------------------------------------------
# 3. Features and model
# ---------------------------------------------------------------------------

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


def cross_val_scores(df, folds=5):
    """Out-of-fold murmur probability per window and per recording. Cached.

    Shared by every report below, so a chart can never show numbers a different
    run produced.
    """
    X, owner = build_matrix(df)
    y = (df.label == "murmur").to_numpy().astype(int)
    key = hashlib.sha1(f"{''.join(df.fname)}|{folds}|{N_FEATURES}".encode()).hexdigest()
    cache = CACHE_DIR / f"pred-{key[:16]}.npz"
    if cache.exists():
        c = np.load(cache)
        return c["wprob"], owner, c["prob"]

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced", random_state=0)
    wprob = cross_val_predict(
        clf, X, y[owner], groups=df.patient.to_numpy()[owner],
        cv=GroupKFold(n_splits=folds), method="predict_proba", n_jobs=-1)[:, 1]
    # Window votes -> recording score. Mean, not max: one noisy window should not
    # condemn a probe.
    prob = np.bincount(owner, weights=wprob) / np.bincount(owner)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, wprob=wprob, prob=prob)
    return wprob, owner, prob


def screening_threshold(truth, prob, target_recall):
    """Lowest threshold reaching the target recall, at this fusion level."""
    fpr, tpr, thr = roc_curve(truth, prob)
    return float(thr[int(np.argmax(tpr >= target_recall))])


def balanced_threshold(truth, prob):
    """Youden J -- the operating point every report quotes. Anything shown to a
    clinician must use this same one, or the case cards and the headline figures
    describe two different machines."""
    fpr, tpr, thr = roc_curve(truth, prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def train_model(df, folds=5, path=MODEL):
    """Fit on every recording and save the model with its decision threshold.

    Cross-validation measures performance but throws each fold's model away, so
    it cannot score a new recording. This fits one final model on all the data
    for deployment. The threshold is carried WITH the model deliberately: a
    score means nothing without the cut-off it was tuned against, and shipping
    them separately is how a deployed screener silently changes its mind.

    The reported AUC still comes from cross-validation, never from this fit --
    a model scored on its own training data would look far better than it is.
    """
    X, owner = build_matrix(df)
    y = (df.label == "murmur").to_numpy().astype(int)
    _, _, prob = cross_val_scores(df, folds)
    thr = balanced_threshold(y, prob)

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced", random_state=0)
    clf.fit(X, y[owner])

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "threshold": thr, "n_features": N_FEATURES, "sr": SR,
                 "trained_on": f"{len(df)} recordings, {df.patient.nunique()} patients",
                 "auc_crossval": float(roc_auc_score(y, prob))}, path)
    print(f"wrote {path}\n  threshold {thr:.3f} | cross-validated AUC "
          f"{roc_auc_score(y, prob):.3f} · trained on {len(df)} recordings")
    return path


def _fit_and_score(tr, te):
    """Train on one corpus, score another. Returns per-recording probabilities."""
    Xtr, otr = build_matrix(tr)
    Xte, ote = build_matrix(te)
    ytr = (tr.label == "murmur").to_numpy().astype(int)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr[otr])
    w = clf.predict_proba(Xte)[:, 1]
    return np.bincount(ote, weights=w) / np.bincount(ote)


def transfer_test(train_sets=("circor",), folds=5):
    """Does the model work on a corpus it has never seen?

    A cross-validated score says how well the model does on MORE OF THE SAME
    data -- same hardware, same population, same clinic. It cannot say whether
    the thing works anywhere else, and "anywhere else" is where a wearable ends
    up. Two held-out corpora are scored here, and they answer different
    questions, so the gap between them is the point:

      PASCAL set_b  labels "is a murmur audible" -- the SAME target CirCor uses,
                    on a different collection. This is the real generalisation
                    test, and the model holds up: 0.822 in-domain -> 0.799.

      CinC 2016     labels "does this patient have a confirmed cardiac
                    diagnosis", and its abnormal group explicitly includes
                    CORONARY ARTERY DISEASE, which produces no murmur. The model
                    scores 0.549 here -- near chance -- which is not a
                    generalisation failure but empirical confirmation of the
                    scope boundary this project has claimed all along: it hears
                    valves, not arteries. Quote it as evidence for that limit,
                    never as this model's performance.

    A warning about CinC that cost real time to find: pooled cross-validation on
    it scores 0.973, which is not a result. Its six sub-databases are 96%
    identifiable from audio alone and carry abnormal rates from 8.5% to 77%, so
    a model can score well by recognising the recording device and reciting that
    device's base rate. Never train or cross-validate on pooled CinC.
    """
    tr = load_index(sets=train_sets)
    ytr = (tr.label == "murmur").to_numpy().astype(int)
    _, _, prob_in = cross_val_scores(tr, folds)
    auc_in = roc_auc_score(ytr, prob_in)
    print(f"train: {len(tr)} recordings, {tr.patient.nunique()} patients "
          f"({', '.join(train_sets)})")
    print(f"\n  in-domain (cross-validated, split by patient)   ROC-AUC {auc_in:.3f}")

    for sets, name, note in (
            (("b",), "PASCAL set_b", "same label definition -- the real test"),
            (("cinc",), "CinC 2016", "DIFFERENT target: includes coronary disease")):
        try:
            te = load_index(sets=sets)
        except FileNotFoundError as e:
            print(f"\n  {name}: skipped ({e})")
            continue
        yte = (te.label == "murmur").to_numpy().astype(int)
        prob = _fit_and_score(tr, te)
        auc = roc_auc_score(yte, prob)
        fpr, tpr, thr = roc_curve(yte, prob)
        i = int(np.argmax(tpr - fpr))
        print(f"\n  -> {name}: {len(te)} recordings, {int(yte.sum())} positive")
        print(f"     {note}")
        print(f"     TRANSFERRED ROC-AUC {auc:.3f}   (drop {auc_in - auc:+.3f})")
        print(f"     at balanced point: recall {tpr[i]:.1%} | "
              f"specificity {1 - fpr[i]:.1%}")
        if te.dataset.nunique() > 1:
            d = te.assign(p=prob, t=yte)
            for s, g in d.groupby("dataset"):
                n = (f"AUC {roc_auc_score(g.t, g.p):.3f}" if g.t.nunique() > 1
                     else "single class")
                print(f"       {s:10s} n={len(g):5d}  positive={int(g.t.sum()):4d}  {n}")


def predict(wav_paths, path=MODEL):
    """Score new recordings with a saved model. This is the deployment path.

    Prints one line per file. A flag means "this recording contains sounds
    consistent with a murmur -- refer for an echocardiogram", never a diagnosis.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"no model at {path}. Train one first:\n  python {Path(__file__).name} --train")
    bundle = joblib.load(path)
    clf, thr = bundle["model"], bundle["threshold"]
    if bundle.get("n_features") != N_FEATURES:
        raise ValueError(
            f"model expects {bundle.get('n_features')} features, this code produces "
            f"{N_FEATURES}. Retrain with --train.")

    print(f"model: {path.name} | threshold {thr:.3f} | {bundle.get('trained_on', '?')}\n")
    results = []
    for p in wav_paths:
        p = Path(p)
        ws = windows(load_audio(p))
        scores = clf.predict_proba(np.vstack([features(w) for w in ws]))[:, 1]
        score = float(scores.mean())          # same fusion the reports use
        flag = score > thr
        loudest = float(scores.max())
        print(f"  {p.name:<34} score {score:.3f}  "
              f"{'MURMUR DETECTED' if flag else 'clear':<16} "
              f"(loudest window {loudest:.3f}, {len(ws)} windows)")
        results.append({"file": str(p), "score": score, "flag": bool(flag)})
    print("\nA flag means: refer for an echocardiogram. It is not a diagnosis,\n"
          "and it says nothing about coronary artery disease.")
    return results


# ---------------------------------------------------------------------------
# 3. Text report
# ---------------------------------------------------------------------------

def _curve(title, truth, prob, target_recall):
    """AUC plus the operating points that matter, each with its OWN threshold.

    A threshold tuned on single recordings is wrong for fused patient scores --
    max-of-four-probes shifts the whole distribution up, so reusing it floods the
    patient report with false alarms. Every level re-tunes.
    """
    auc = roc_auc_score(truth, prob)
    fpr, tpr, thr = roc_curve(truth, prob)
    print(f"\n== {title}  (n={len(truth)}, {int(truth.sum())} murmur)   ROC-AUC {auc:.3f} ==")
    print(f"  {'operating point':<22}{'thresh':>8}{'recall':>9}{'specificity':>13}{'precision':>11}")
    picks = [("balanced (Youden J)", int(np.argmax(tpr - fpr)))]
    for r in (0.80, target_recall):
        picks.append((f"recall >= {r:.0%}", int(np.argmax(tpr >= r))))
    best = thr[picks[-1][1]]
    for name, i in picks:
        pred = prob > thr[i]
        prec = pred[truth == 1].sum() / max(pred.sum(), 1)
        print(f"  {name:<22}{thr[i]:>8.3f}{tpr[i]:>9.1%}{1 - fpr[i]:>13.1%}{prec:>11.1%}")
    print("  confusion at the screening point (rows true, cols predicted):")
    print("   ", str(confusion_matrix(truth, prob > best)).replace("\n", "\n    "))
    return auc


def text_report(df, prob, target_recall):
    truth = (df.label == "murmur").to_numpy().astype(int)
    _curve("Recording level (one probe)", truth, prob, target_recall)
    d = df.assign(p=prob, t=truth)
    # The jacket has 7-8 probes on one chest, so the verdict that matters is per
    # patient. Max says "any probe that heard something condemns the patient" --
    # right when a murmur is audible at only one site. Report both, let AUC decide.
    for how in ("max", "mean"):
        pat = d.groupby("patient").agg(p=("p", how), t=("t", "max"))
        _curve(f"Patient level, {how} over probes", pat.t.to_numpy(),
               pat.p.to_numpy(), target_recall)
    if df.dataset.nunique() > 1:
        print("\n== Per source set ==")
        for s, g in d.groupby("dataset"):
            note = (f"AUC={roc_auc_score(g.t, g.p):.3f}" if g.t.nunique() > 1
                    else "(single class)")
            print(f"  {s:10s} n={len(g):5d}  murmurs={int(g.t.sum()):4d}  {note}")


def export_detections(df, prob, threshold, path):
    """Every recording ranked by score, with the verdict at this threshold."""
    truth = (df.label == "murmur").to_numpy().astype(int)
    flagged = prob > threshold
    out = df[["fname", "patient", "position"]].copy()
    out["score"] = prob.round(4)
    out["verdict"] = np.where(flagged, "MURMUR DETECTED", "clear")
    out["reference"] = df.label.to_numpy()
    out["outcome"] = np.where(flagged, np.where(truth == 1, "true positive", "false alarm"),
                              np.where(truth == 1, "missed", "true negative"))
    out = out.sort_values("score", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {path}: {len(out)} recordings, {int(flagged.sum())} flagged at "
          f"threshold {threshold:.3f} ({int((truth[flagged] == 1).sum())} of them real)")


# ---------------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------------

def score_over_time(win_scores, duration, n=500):
    """Per-window scores -> a continuous trace, averaging the 50% overlap."""
    t = np.linspace(0, duration, n)
    total, count = np.zeros(n), np.zeros(n)
    for j, p in enumerate(win_scores):
        m = (t >= j * 2) & (t <= j * 2 + 4)
        total[m] += p
        count[m] += 1
    flat = float(np.mean(win_scores))
    if not count.any():
        return t, np.full(n, flat)
    return t, np.where(count > 0, total / np.maximum(count, 1), flat)


def cohort_chart(df, prob, target_recall, path):
    truth = (df.label == "murmur").to_numpy().astype(int)
    d = df.assign(p=prob, t=truth)
    pat = d.groupby("patient").agg(p=("p", "max"), t=("t", "max"))
    thr = screening_threshold(truth, prob, target_recall)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    fig.suptitle("Murmur screener — validation\n"
                 f"{len(df)} recordings · {df.patient.nunique()} patients · "
                 f"{truth.sum()} with an audible murmur · 5-fold, split by patient",
                 x=0.06, ha="left", fontsize=12, color=INK)

    ax = axes[0, 0]                                            # ROC
    for name, tr, pr, colour in (("per probe", truth, prob, BLUE),
                                 ("per patient", pat.t.to_numpy(), pat.p.to_numpy(), ORANGE)):
        fpr, tpr, _ = roc_curve(tr, pr)
        ax.plot(fpr, tpr, lw=2, color=colour, label=f"{name}  AUC {roc_auc_score(tr, pr):.3f}")
    ax.plot([0, 1], [0, 1], lw=1, color=AXIS)
    ax.text(0.55, 0.47, "coin flip", color=MUTED, fontsize=8, rotation=33)
    ax.set(xlabel="false alarms among healthy", ylabel="murmurs caught",
           title="Detection vs false alarms", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(True, lw=0.8); ax.set_axisbelow(True)

    ax = axes[0, 1]                                            # score separation
    bins = np.linspace(0, 1, 31)
    for lab, colour, name in ((0, BLUE, "no murmur"), (1, ORANGE, "murmur")):
        v = prob[truth == lab]
        ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=colour, alpha=0.20)
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2, color=colour,
                label=f"{name}  (n={len(v)})")
    ax.axvline(thr, lw=1.5, color=INK2)
    ax.text(thr + 0.015, ax.get_ylim()[1] * 0.94, f"threshold {thr:.2f}", color=INK2, fontsize=8.5)
    ax.set(xlabel="model score", ylabel="share of recordings",
           title="Score separation (overlap = missed cases)", xlim=(0, 1))
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, lw=0.8); ax.set_axisbelow(True)

    ax = axes[1, 0]                                            # confusion
    cm = confusion_matrix(truth, prob > thr)
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap=BLUES, vmin=0, vmax=1)
    labels = [["true negative", "false alarm"], ["MISSED MURMUR", "caught"]]
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm[i].sum()
            ink = SURFACE if frac > 0.5 else INK
            ax.text(j, i - 0.10, f"{cm[i, j]}", ha="center", va="center", fontsize=17, color=ink)
            ax.text(j, i + 0.18, f"{labels[i][j]}  {frac:.0%}", ha="center", va="center",
                    fontsize=8, color=SURFACE if frac > 0.5 else INK2)
    ax.set(xticks=[0, 1], yticks=[0, 1], title=f"At {target_recall:.0%}-recall threshold",
           xticklabels=["screened out", "sent for echo"], yticklabels=["no murmur", "murmur"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax = axes[1, 1]                                            # calibration
    edges = np.quantile(prob, np.linspace(0, 1, 9)); edges[-1] += 1e-9
    idx = np.clip(np.digitize(prob, edges) - 1, 0, len(edges) - 2)
    xs = [prob[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()]
    ys = [truth[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()]
    ax.plot([0, 1], [0, 1], lw=1, color=AXIS)
    ax.text(0.24, 0.40, "perfectly calibrated", color=MUTED, fontsize=8, rotation=33)
    ax.plot(xs, ys, lw=2, color=BLUE, marker="o", ms=6, markerfacecolor=BLUE,
            markeredgecolor=SURFACE, markeredgewidth=2)
    # Runs below the diagonal because class_weight='balanced' inflates the positive
    # score. Ranking is unaffected (that is what AUC measures) but the number is
    # NOT a probability -- say so rather than let a clinician read 0.40 as "40%".
    ax.set(xlabel="score the model gave", ylabel="how often it was really a murmur",
           title="Calibration — ranks well, reads high", xlim=(0, 1), ylim=(0, 1))
    ax.text(0.03, 0.92, "scores sit above the true rate:\ncompare to a threshold, "
            "do not read as a probability", fontsize=8, color=INK2, va="top")
    ax.grid(True, lw=0.8); ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def patient_chart(df, wprob, owner, patient, threshold, path):
    """One page per patient: every probe, waveform over spectrogram, score trace.

    The score trace is a SEPARATE axis, not a second y-scale on the waveform --
    amplitude and probability share no units, and a dual-axis plot would invent an
    alignment between them.
    """
    rows = df[df.patient == patient]
    n = len(rows)

    # Pass 1: compute every spectrogram, then set ONE dB window for the page from
    # the pooled distribution. A fixed -60 dB window anchored on the loudest bin
    # flattens these into a single mid tone and the S1/S2 structure a clinician
    # reads disappears. Shared limits keep probes comparable with each other.
    panels = []
    for r in rows.itertuples():
        yb = sosfiltfilt(SOS, load_audio(r.path))
        yb = np.ascontiguousarray(yb / (np.abs(yb).max() + 1e-9), dtype=np.float32)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(yb, n_fft=512, hop_length=64)),
                                    ref=np.max)
        panels.append((r, yb, D))
    keep = librosa.fft_frequencies(sr=SR, n_fft=512) <= 800
    vmin, vmax = np.percentile(np.concatenate([D[keep].ravel() for _, _, D in panels]),
                               [20, 99.7])

    fig = plt.figure(figsize=(11, 1.0 + 2.5 * n))
    gs = fig.add_gridspec(n * 3, 1, height_ratios=[0.7, 2.2, 0.35] * n, hspace=0.0,
                          left=0.09, right=0.90, top=1 - 0.55 / (1.0 + 2.5 * n), bottom=0.05)

    truth = "murmur present" if (rows.label == "murmur").any() else "no murmur"
    flagged = [r.position for r in rows.itertuples()
               if wprob[owner == r.Index].mean() > threshold]
    verdict = f"FLAGGED at {', '.join(flagged)}" if flagged else "clear at every probe"
    fig.suptitle(f"Patient {patient}   —   screener says: {verdict}\n"
                 f"reference standard: {truth}   ·   threshold {threshold:.3f}   ·   "
                 f"scores are out-of-fold (this patient was never in training)",
                 x=0.09, ha="left", fontsize=11, color=INK, y=0.995)

    for k, (r, yb, D) in enumerate(panels):
        dur = len(yb) / SR
        ws = wprob[owner == r.Index]
        rec_score = float(ws.mean())
        hit = rec_score > threshold

        ax_w = fig.add_subplot(gs[k * 3])
        ax_s = fig.add_subplot(gs[k * 3 + 1], sharex=ax_w)
        ax_p = fig.add_subplot(gs[k * 3 + 2], sharex=ax_w)

        # Scaled to the 99.5th percentile, not the peak: a single handling click
        # otherwise flattens the whole trace. Display only -- features use peak.
        disp = yb / (np.percentile(np.abs(yb), 99.5) + 1e-9)
        ax_w.plot(np.arange(len(yb)) / SR, np.clip(disp, -1, 1), lw=0.4, color=BLUE)
        ax_w.set_ylim(-1.15, 1.15); ax_w.set_yticks([])
        ax_w.tick_params(labelbottom=False)
        for s in ("left", "bottom"):
            ax_w.spines[s].set_visible(False)
        ax_w.set_ylabel(f"{r.position}", rotation=0, ha="right", va="center",
                        fontsize=13, color=INK, labelpad=14)
        ax_w.text(1.006, 0.5, ("●  FLAG" if hit else "●  clear"), transform=ax_w.transAxes,
                  color=CRITICAL if hit else GOOD, fontsize=9, va="center")
        ax_w.text(1.006, -0.35, f"score {rec_score:.2f}", transform=ax_w.transAxes,
                  color=INK2, fontsize=9, va="center")

        im = ax_s.imshow(D[keep], origin="lower", aspect="auto", cmap=BLUES,
                         extent=[0, dur, 0, 800], vmin=vmin, vmax=vmax,
                         interpolation="nearest")
        ax_s.set_ylabel("Hz", color=INK2); ax_s.set_yticks([0, 400, 800])
        ax_s.tick_params(labelbottom=False)

        t, trace = score_over_time(ws, dur)
        ax_p.imshow(trace[None, :], aspect="auto", cmap=DIVERGING,
                    norm=TwoSlopeNorm(vmin=0.0, vcenter=max(threshold, 1e-3), vmax=1.0),
                    extent=[0, dur, 0, 1], interpolation="bilinear")
        ax_p.set_yticks([]); ax_p.set_xlim(0, dur)
        ax_p.set_ylabel("model", rotation=0, ha="right", va="center",
                        fontsize=8, color=MUTED, labelpad=14)
        if k == n - 1:
            ax_p.set_xlabel("seconds", color=INK2)
        else:
            ax_p.tick_params(labelbottom=False)

        if k == 0:
            cb = fig.colorbar(im, ax=[ax_w, ax_s, ax_p], fraction=0.02, pad=0.10)
            cb.set_label("dB below peak", color=INK2, fontsize=8)
            cb.outline.set_visible(False)
            cb.ax.tick_params(color=MUTED, labelcolor=MUTED, labelsize=7)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Audio cards
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("caught", "murmur, caught", "good",
     "The reference standard says murmur and the screener agrees. Listen for a sustained "
     "rasp filling the space after the first heart sound."),
    ("missed", "murmur, missed", "critical",
     "A murmur the screener let through. Judge whether it was audible at all, or whether "
     "handling noise buried it -- the two failures need different fixes."),
    ("falsealarm", "false alarm", "warning",
     "Flagged, but the reference standard records no murmur. Decide whether what fills "
     "systole is turbulence or friction against the sensor."),
    ("clear", "clear", "good",
     "Two crisp sounds per cycle with silence between them. This is the baseline every "
     "other card should be compared against."),
]


def _excerpt(y):
    """Centred window. The ends of a recording hold probe placement and removal."""
    n = CLIP_SECONDS * SR
    if len(y) <= n:
        return y, 0.0
    start = (len(y) - n) // 2
    return y[start:start + n], start / SR


def _playable(y):
    """Bandpass and normalise so it is audible on laptop speakers. Level comes from
    the 99.5th percentile, not the peak: one handling click would otherwise drag the
    whole clip down to inaudible."""
    y = sosfiltfilt(PLAYBACK_SOS, y)
    return np.clip(y / (np.percentile(np.abs(y), 99.5) + 1e-9) * 0.7, -1, 1)


def _mp3(y):
    """Encode to mp3. Source is 4 kHz so all content is under 2 kHz and 64 kbps is
    transparent for it; mp3 rather than opus purely for universal playback."""
    with tempfile.TemporaryDirectory() as tmp:
        wav, mp3 = Path(tmp) / "a.wav", Path(tmp) / "a.mp3"
        sf.write(wav, y.astype(np.float32), SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav),
                        "-ac", "1", "-b:a", "64k", "-ar", "8000", str(mp3)], check=True)
        return "data:audio/mpeg;base64," + base64.b64encode(mp3.read_bytes()).decode()


def _mini_chart(y, win_scores, threshold, duration):
    """Small spectrogram + score strip for exactly the excerpt being played."""
    fig = plt.figure(figsize=(6.4, 1.85))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 0.4], hspace=0.08,
                          left=0.075, right=0.995, top=0.985, bottom=0.20)
    ax_s, ax_p = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=64)), ref=np.max)
    keep = librosa.fft_frequencies(sr=SR, n_fft=512) <= 800
    vmin, vmax = np.percentile(D[keep], [20, 99.7])
    ax_s.imshow(D[keep], origin="lower", aspect="auto", cmap=BLUES,
                extent=[0, duration, 0, 800], vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_s.set_yticks([0, 400, 800]); ax_s.set_ylabel("Hz", color=INK2, fontsize=7.5)
    ax_s.tick_params(labelbottom=False, labelsize=7, colors=MUTED)

    _, trace = score_over_time(win_scores, duration)
    ax_p.imshow(trace[None, :], aspect="auto", cmap=DIVERGING,
                norm=TwoSlopeNorm(vmin=0.0, vcenter=max(threshold, 1e-3), vmax=1.0),
                extent=[0, duration, 0, 1], interpolation="bilinear")
    ax_p.set_yticks([]); ax_p.set_xlabel("seconds into the clip", color=INK2, fontsize=7.5)
    ax_p.tick_params(labelsize=7, colors=MUTED)
    for ax in (ax_s, ax_p):
        for s in ax.spines.values():
            s.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_cards(df, wprob, owner, prob, threshold, per_category=1):
    """One group per clinical outcome; inside it, every probe of a chosen patient."""
    truth = (df.label == "murmur").to_numpy().astype(int)
    d = df.assign(p=prob, t=truth)
    pat = d.groupby("patient").agg(p=("p", "max"), t=("t", "max"), n=("p", "size"))
    pools = {
        "caught": pat[(pat.t == 1) & (pat.p > threshold)].sort_values("p", ascending=False),
        "missed": pat[(pat.t == 1) & (pat.p <= threshold)].sort_values("p"),
        "falsealarm": pat[(pat.t == 0) & (pat.p > threshold)].sort_values("p", ascending=False),
        "clear": pat[(pat.t == 0) & (pat.p <= threshold)].sort_values("p"),
    }
    # Prefer patients recorded at several sites -- a single-probe case gives a
    # clinician nothing to compare across the chest, which is half the point.
    pools = {k: (v[v.n >= 3] if (v.n >= 3).any() else v) for k, v in pools.items()}

    groups = []
    for kind, label, tone, guidance in CATEGORIES:
        cards = []
        for pid in pools[kind].index[:per_category]:
            for r in d[d.patient == pid].itertuples():
                y = load_audio(r.path)
                clip, offset = _excerpt(y)
                play = _playable(clip)
                dur = len(clip) / SR
                ws = wprob[owner == r.Index]
                lo = int(offset // 2)
                sel = ws[lo:lo + max(1, int(dur // 2))] if len(ws) > 1 else ws
                cards.append({
                    "patient": pid, "site": r.position, "score": float(d.p[r.Index]),
                    "truth": r.label, "flag": float(d.p[r.Index]) > threshold,
                    "excerpt": "" if len(y) <= CLIP_SECONDS * SR
                               else f"{offset:.0f}–{offset + dur:.0f}s of {len(y) / SR:.0f}s",
                    "audio": _mp3(play),
                    "image": _mini_chart(play, sel if len(sel) else ws, threshold, dur),
                })
                print(f"  {kind}: {pid} {r.position}", flush=True)
        groups.append((kind, label, tone, guidance, cards))
    return groups


# ---------------------------------------------------------------------------
# 6. Review page
# ---------------------------------------------------------------------------

STYLE = """
  :root {
    color-scheme: light;
    --plane:#f4f4f1; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
    --muted:#898781; --rule:#e1e0d9; --accent:#2a78d6;
    --good:#0ca30c; --warning:#b8791a; --critical:#d03b3b;
    --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: ui-monospace, "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
      --muted:#898781; --rule:#2c2c2a; --accent:#3987e5;
      --good:#0ca30c; --warning:#fab219; --critical:#e66767;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
    --muted:#898781; --rule:#2c2c2a; --accent:#3987e5;
    --good:#0ca30c; --warning:#fab219; --critical:#e66767;
  }
  body { background: var(--plane); color: var(--ink); font-family: var(--sans);
         line-height: 1.6; margin: 0;
         padding: clamp(1.5rem,4vw,4rem) clamp(1rem,4vw,2rem) 6rem; }
  main { max-width: 80rem; margin: 0 auto; display: flex; flex-direction: column; gap: 3.5rem; }
  p, li { max-width: 68ch; color: var(--ink-2); }
  h1,h2,h3 { text-wrap: balance; color: var(--ink); margin: 0; }
  h1 { font-size: clamp(1.9rem,4vw,2.6rem); font-weight: 650; letter-spacing:-.022em; }
  h2 { font-size: 1.35rem; font-weight: 620; letter-spacing:-.012em; }
  h3 { font-size: 1rem; font-weight: 620; }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
  .eyebrow { font-family: var(--mono); font-size:.72rem; letter-spacing:.14em;
             text-transform: uppercase; color: var(--muted); margin: 0 0 .6rem; }
  header { display: flex; flex-direction: column; gap: 1rem; }
  .lede { font-size: 1.06rem; color: var(--ink-2); }
  section { display: flex; flex-direction: column; gap: 1.25rem; }
  .sec-head { border-top: 1px solid var(--rule); padding-top: 1.25rem; }
  .metrics { display: flex; flex-wrap: wrap; gap: 2.5rem; }
  .metric { display: flex; flex-direction: column; gap: .1rem; }
  .metric b { font-family: var(--mono); font-size: 2rem; font-weight: 600;
              color: var(--ink); line-height: 1.1; }
  .metric span { font-size: .82rem; color: var(--muted); }
  .split { display: grid; grid-template-columns: repeat(auto-fit,minmax(19rem,1fr)); gap: 1rem; }
  .panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
           padding: 1.25rem 1.4rem; display: flex; flex-direction: column; gap: .5rem; }
  .panel p { margin: 0; font-size: .94rem; }
  .panel--yes { border-left: 3px solid var(--good); }
  .panel--no  { border-left: 3px solid var(--critical); }
  .plain { background: var(--surface); border: 1px solid var(--rule);
           border-left: 3px solid var(--accent); border-radius: 3px;
           padding: 1.5rem 1.75rem; display: flex; flex-direction: column; gap: 1rem; }
  .plain p { font-size: 1.05rem; max-width: 62ch; margin: 0; }
  .plain p strong { color: var(--ink); }
  .caveat { font-size: .95rem; color: var(--ink-2); border-top: 1px solid var(--rule);
            padding-top: 1rem; margin: 0; max-width: 62ch; }
  /* Diagnostic images are read on a lightbox. They stay light in both themes --
     inverting a spectrogram for dark mode invites a misread. */
  .plate { margin: 0; background: #fcfcfb; border: 1px solid var(--rule);
           border-radius: 3px; padding: .75rem; overflow-x: auto; }
  .plate img { display: block; width: 100%; height: auto; min-width: 40rem; }
  table { border-collapse: collapse; width: 100%; font-size: .9rem; }
  caption { text-align: left; color: var(--muted); font-size: .82rem; padding-bottom: .6rem; }
  th, td { text-align: right; padding: .5rem .9rem; border-bottom: 1px solid var(--rule);
           font-variant-numeric: tabular-nums; }
  thead th { font-family: var(--mono); font-size: .7rem; letter-spacing:.1em;
             text-transform: uppercase; color: var(--muted); font-weight: 500; }
  tbody th { text-align: left; color: var(--ink-2); font-weight: 500; }
  td:first-of-type { text-align: left; color: var(--ink-2); }
  .table-wrap { overflow-x: auto; }
  .group { display: flex; flex-direction: column; gap: 1rem; }
  .group-head { display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap; }
  .chip { font-family: var(--mono); font-size: .72rem; letter-spacing:.08em;
          text-transform: uppercase; padding: .2rem .55rem; border-radius: 2px;
          border: 1px solid currentColor; }
  .chip--good { color: var(--good); }
  .chip--warning { color: var(--warning); }
  .chip--critical { color: var(--critical); }
  .cards { display: grid; grid-template-columns: repeat(auto-fit,minmax(26rem,1fr)); gap: 1rem; }
  .card { background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
          padding: .9rem 1rem 1rem; display: flex; flex-direction: column; gap: .6rem; }
  .card-head { display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap; }
  .site { font-family: var(--mono); font-size: 1.05rem; font-weight: 600; color: var(--ink); }
  .card-meta { font-family: var(--mono); font-size: .76rem; color: var(--muted);
               margin-left: auto; font-variant-numeric: tabular-nums; }
  .verdict { font-family: var(--mono); font-size: .7rem; letter-spacing:.07em;
             text-transform: uppercase; }
  .verdict--flag { color: var(--critical); }
  .verdict--clear { color: var(--good); }
  .card audio { width: 100%; height: 34px; }
  .card .shot { display: block; width: 100%; height: auto; border-radius: 2px;
                background: #fcfcfb; }
  .legend { display: grid; grid-template-columns: repeat(auto-fit,minmax(15rem,1fr)); gap: 1rem; }
  .legend div { display: flex; flex-direction: column; gap: .15rem; }
  .legend b { font-size: .9rem; color: var(--ink); }
  .legend span { font-size: .88rem; color: var(--ink-2); }
  footer { border-top: 1px solid var(--rule); padding-top: 1.25rem;
           color: var(--muted); font-size: .85rem; }
  code { font-family: var(--mono); font-size: .85em; }
"""


def _plain_numbers(df, prob, threshold):
    """The figures the page leads with, computed from the same run it displays."""
    truth = (df.label == "murmur").to_numpy().astype(int)
    pat = df.assign(p=prob, t=truth).groupby("patient").agg(p=("p", "max"), t=("t", "max"))
    y, s = pat.t.to_numpy(), pat.p.to_numpy()
    thr = balanced_threshold(y, s)
    pred = s > thr
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    return {
        "found": round(tp / max(tp + fn, 1) * 100),
        "cleared": round(tn / max(tn + fp, 1) * 100),
        "precision": round(tp / max(tp + fp, 1) * 100),
        "patients": len(y),
        "accuracy": round((tp + tn) / len(y) * 100),
        "baseline": round((y == 0).mean() * 100),
        "auc_probe": roc_auc_score(truth, prob),
        "auc_patient": roc_auc_score(y, s),
        "recordings": len(df),
        "murmurs": int(truth.sum()),
    }


def _card_html(cards):
    out = []
    for c in cards:
        verdict = ("flagged", "flag") if c["flag"] else ("clear", "clear")
        excerpt = f" &middot; {c['excerpt']}" if c["excerpt"] else ""
        out.append(f"""
        <article class="card">
          <div class="card-head">
            <span class="site">{c['site']}</span>
            <span class="verdict verdict--{verdict[1]}">{verdict[0]}</span>
            <span class="card-meta">score {c['score']:.2f} &middot; reference: {c['truth']}{excerpt}</span>
          </div>
          <audio controls preload="metadata" src="{c['audio']}">
            Your browser cannot play audio. The spectrogram below shows the same clip.
          </audio>
          <img class="shot" src="{c['image']}"
               alt="Spectrogram and model score for probe {c['site']}, patient {c['patient']}">
        </article>""")
    return "\n".join(out)


def render_page(groups, threshold, stats, operating, cohort_png, path):
    cohort_img = ("data:image/png;base64," + base64.b64encode(cohort_png.read_bytes()).decode()
                  if cohort_png.exists() else "")
    rows = "\n".join(
        f'<tr><th scope="row">{lvl}</th><td>{op}</td><td>{r}</td><td>{s}</td><td>{p}</td></tr>'
        for lvl, pts in operating for op, r, s, p in pts)
    blocks = "\n".join(f"""
      <div class="group">
        <div class="group-head">
          <span class="chip chip--{tone}">{label}</span>
          <span class="card-meta">patient {cards[0]['patient'] if cards else '—'} &middot; {len(cards)} probes</span>
        </div>
        <p>{guidance}</p>
        <div class="cards">{_card_html(cards)}</div>
      </div>""" for kind, label, tone, guidance, cards in groups)
    cohort_block = (f'<figure class="plate"><img src="{cohort_img}" alt="Four validation '
                    'charts: ROC curve, score separation by true class, confusion matrix at '
                    'the screening threshold, and calibration curve"></figure>'
                    if cohort_img else "")
    u = '<span style="font-size:1rem">'

    html = f"""<title>Murmur Screener Review</title>
<style>{STYLE}</style>

<main>
  <header>
    <p class="eyebrow">Vital Jacket &middot; Phase 1 &middot; digital stethoscope</p>
    <h1>Murmur screener &mdash; listen and confirm</h1>
    <p class="lede">
      An acoustic screener that listens to each chest probe and flags recordings containing a
      murmur. Every clip below is playable, paired with the spectrogram of that same excerpt.
      Scores are out-of-fold: no patient was in the model's training data when it was scored.
      Please confirm or overrule the calls.
    </p>
    <div class="metrics">
      <div class="metric"><b>{stats['found']}{u}&thinsp;in&thinsp;100</span></b><span>murmurs it finds</span></div>
      <div class="metric"><b>{stats['cleared']}{u}&thinsp;in&thinsp;100</span></b><span>healthy people it clears</span></div>
      <div class="metric"><b>{stats['precision']}{u}%</span></b><span>of those it flags really have one</span></div>
      <div class="metric"><b>{stats['patients']}</b><span>patients tested</span></div>
    </div>
  </header>

  <section>
    <div class="plain">
      <h2>In plain terms</h2>
      <p>This is a <strong>listening test, not a scan</strong>. It listens to the sound a heart
        makes and answers one question: does this heart sound abnormal enough that someone
        should get an ultrasound?</p>
      <p>It can hear a <strong>blocked or leaking heart valve</strong>. When a valve is
        narrowed, blood is forced through a small opening and rushes &mdash; that hiss is a
        murmur, and it is what the model listens for.</p>
      <p>It <strong>cannot hear a blocked artery</strong> &mdash; the kind of blockage that
        causes heart attacks. Those are effectively silent to it. It also does not detect
        &ldquo;heart block&rdquo; in the cardiology sense, which is an electrical fault found
        on an ECG, not a sound. Nothing here says anything about anyone's arteries.</p>
      <p>Set the way we recommend: out of every 100 people who really have a murmur it finds
        <strong>{stats['found']}</strong>, and out of every 100 healthy people it correctly
        clears <strong>{stats['cleared']}</strong>. Of the people it does flag, about
        <strong>{stats['precision']}%</strong> really do have a murmur. It is a first filter,
        not a diagnosis &mdash; a doctor confirms every flag with a proper scan.</p>
      <p class="caveat"><strong>On &ldquo;{stats['accuracy']}% accurate&rdquo;.</strong> You can
        quote that figure and it is true, but it flatters the tool. Roughly
        {stats['baseline']} in 100 people tested are healthy, so a machine that simply said
        &ldquo;healthy&rdquo; to everyone and listened to nothing would already score
        {stats['baseline']}%. The honest gain is {stats['accuracy']}% against that
        {stats['baseline']}% floor. The two numbers worth quoting are the ones above.</p>
    </div>
  </section>

  <section>
    <div class="sec-head"><h2>What it detects &mdash; and what it does not</h2></div>
    <div class="split">
      <div class="panel panel--yes">
        <h3>Valve blockage &mdash; detected</h3>
        <p>A narrowed or leaking valve forces blood through a small opening. The resulting
           turbulence is audible, and it is what this model is trained on. Aortic stenosis
           falls here.</p>
      </div>
      <div class="panel panel--no">
        <h3>Coronary artery blockage &mdash; not detected</h3>
        <p>The model says nothing about coronary disease. Training that would require
           recordings labelled against coronary angiography; no open dataset has them.
           Do not read a flag here as a statement about coronary risk.</p>
      </div>
    </div>
    <p>A murmur is a finding, not a diagnosis &mdash; many are innocent. The intended output is
       &ldquo;this chest sounds abnormal, send it for an echocardiogram&rdquo;: a decision about
       who gets imaging, not a verdict.</p>
  </section>

  <section>
    <div class="sec-head"><h2>How to read a card</h2></div>
    <div class="legend">
      <div><b>Site</b><span>Where on the chest the probe sat &mdash; AV, PV, TV, MV.</span></div>
      <div><b>Audio</b><span>A {CLIP_SECONDS}-second excerpt from the middle of the recording,
        bandpassed 20&ndash;1000 Hz and level-matched &mdash; the band an electronic stethoscope
        passes.</span></div>
      <div><b>Spectrogram</b><span>0&ndash;800 Hz of the clip you just heard. Darker is louder.
        Clean hearts show narrow bands under ~300 Hz with quiet gaps; a murmur widens them and
        fills the gaps.</span></div>
      <div><b>Score strip</b><span>The model's score along the clip. Blue below the decision
        threshold ({threshold:.2f}), red above it &mdash; so you can see <em>where</em> it heard
        something.</span></div>
    </div>
  </section>

  <section>
    <div class="sec-head"><h2>Case review</h2></div>
    <p>One patient per clinical outcome, with every probe recorded for them. Both kinds of
       error are included; the failures are the ones worth your time.</p>
    {blocks}
  </section>

  <section>
    <div class="sec-head"><h2>Does it work</h2></div>
    <p>The technical version of the numbers above. Five-fold cross-validation, split by
       patient: <strong>ROC-AUC {stats['auc_probe']:.3f}</strong> per probe and
       <strong>{stats['auc_patient']:.3f}</strong> per patient, across {stats['recordings']}
       recordings of which {stats['murmurs']} carry an audible murmur. Read the trade-off
       rather than any single row &mdash; screening at 90% recall means echoing roughly two
       thirds of healthy patients, which is likely too noisy to run. The plain-language figures
       at the top come from the <em>balanced</em> row.</p>
    {cohort_block}
    <div class="table-wrap">
      <table>
        <caption>The same operating points as numbers. Precision is the share of flagged cases
          that truly had a murmur.</caption>
        <thead><tr><th scope="col">Level</th><th scope="col">Operating point</th>
          <th scope="col">Recall</th><th scope="col">Specificity</th>
          <th scope="col">Precision</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <p><strong>The score is not a probability.</strong> The model is deliberately weighted
       toward the rarer positive class, so its scores run above the true rate. It ranks
       correctly, which is what the threshold relies on, but 0.40 does not mean 40%.</p>
  </section>

  <section>
    <div class="sec-head"><h2>Before you rely on this</h2></div>
    <ul>
      <li><strong>The training population is paediatric.</strong> CirCor is 664 children, 126
        infants, 72 adolescents and 6 neonates from screening campaigns in Brazil. Its murmurs
        are largely innocent or congenital. If the jacket is for adults, whose murmurs are
        typically degenerative aortic stenosis, performance here is unmeasured and probably
        optimistic.</li>
      <li><strong>Recordings are clinical, not ambulatory.</strong> Captured with a stethoscope
        held still by a clinician. A probe worn on a moving person will carry far more motion
        and friction artifact.</li>
      <li><strong>Most murmurs here are faint</strong> (grade I&ndash;II of VI), realistic for
        screening but not easy examples.</li>
      <li><strong>Not a medical device.</strong> Not validated, not certified, not for clinical
        use.</li>
    </ul>
  </section>

  <footer>
    Audio and labels: <strong>The CirCor DigiScope Phonocardiogram Dataset v1.0.3</strong>,
    Oliveira et al., PhysioNet &mdash; de-identified, open access, ODC-By 1.0, redistributed
    with attribution. Generated by <code>vital_jacket.py</code>.
  </footer>
</main>
"""
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def _operating_table(df, prob):
    """Rows for the page's operating-point table, at both fusion levels."""
    truth = (df.label == "murmur").to_numpy().astype(int)
    pat = df.assign(p=prob, t=truth).groupby("patient").agg(p=("p", "max"), t=("t", "max"))
    table = []
    for name, tr, pr in (("Per probe", truth, prob),
                         ("Per patient", pat.t.to_numpy(), pat.p.to_numpy())):
        fpr, tpr, thr = roc_curve(tr, pr)
        pts = []
        for label, i in (("Balanced", int(np.argmax(tpr - fpr))),
                         ("Recall ≥ 80%", int(np.argmax(tpr >= 0.80))),
                         ("Recall ≥ 90%", int(np.argmax(tpr >= 0.90)))):
            pred = pr > thr[i]
            prec = pred[tr == 1].sum() / max(pred.sum(), 1)
            pts.append((label, f"{tpr[i]:.1%}", f"{1 - fpr[i]:.1%}", f"{prec:.1%}"))
        table.append((name, pts))
    return table


# ---------------------------------------------------------------------------
# 7. Self-check
# ---------------------------------------------------------------------------

def self_check():
    """Every invariant that, if broken, silently inflates the scores."""
    assert patient_and_position("set_b/extrastole__127_1306764300147_C2.wav") == ("b127", "C")
    # ONE underscore -- the 149-file noisy subset that breaks split('__').
    assert patient_and_position("set_b/normal_noisynormal_101_1305030823364_B.wav") == ("b101", "B")
    pid, pos = patient_and_position("set_a/normal__201102081321.wav")
    assert pid.startswith("a") and pos == "?"
    print("ok  filename parsing (both conventions)")

    # Zero-padding would fake the quiet gaps between beats and read as healthy.
    beat = np.tile(np.r_[np.ones(200), np.zeros(800)].astype(np.float32), 3)
    w = windows(beat)[0]
    assert len(w) == WIN and np.abs(w[-1000:]).max() > 0, "short clip was zero-padded"
    print("ok  short clips tiled, not silence-padded")

    df = load_index()
    assert df.patient.nunique() < len(df), "expected repeat patients"
    for train, test in GroupKFold(n_splits=5).split(df, groups=df.patient):
        overlap = set(df.patient[train]) & set(df.patient[test])
        assert not overlap, f"patient in both train and test: {sorted(overlap)[:3]}"
    print(f"ok  no patient spans a fold ({df.patient.nunique()} patients, {len(df)} recordings)")

    a = features(windows(load_audio(df.iloc[0].path))[0])
    assert a.shape == (N_FEATURES,) and np.isfinite(a).all()
    print(f"ok  features fixed width ({N_FEATURES}) and finite")

    if CIRCOR.exists():
        meta = pd.read_csv(CIRCOR / "training_data.csv")
        partial = meta[(meta.Murmur == "Present") & meta.apply(
            lambda r: 0 < len(set(str(r["Murmur locations"]).split("+"))) <
                      len(set(str(r["Recording locations:"]).split("+"))), axis=1)]
        assert len(partial), "expected partially-audible murmur patients"
        r = partial.iloc[0]
        heard = set(str(r["Murmur locations"]).split("+"))
        got = df[df.patient == f"c{r['Patient ID']}"]
        for row in got.itertuples():
            want = "murmur" if row.position in heard else "normal"
            assert row.label == want, f"{row.fname}: {row.label}, site-truth {want}"
        assert (got.label == "normal").any()
        print("ok  CirCor labelled by site, not by patient")

    # The five hand-built cues are the model's physiological anchor. Verify on real
    # labelled audio, in each cue's own units -- duty is a fraction, rest are dB.
    src = "b" if (PASCAL / "set_b.csv").exists() else "circor"
    dfc = load_index(sets=(src,))
    means = {}
    for lab in ("normal", "murmur"):
        sub = dfc[dfc.label == lab].sample(40, random_state=0)
        means[lab] = np.mean([features(windows(load_audio(p))[0])[-5:] for p in sub.path], axis=0)
    sep = means["murmur"] - means["normal"]
    for (name, want, margin), got in zip(
            [("brightness", +1, .5), ("duty", +1, .03), ("non-peak brightness", +1, .5),
             ("dynamic range", +1, .5), ("HF gaps-vs-peaks", -1, .5)], sep):
        assert np.sign(got) == want and abs(got) > margin, \
            f"cue '{name}' moved {got:+.3f}, expected sign {want:+d} margin {margin}"
    print("ok  murmur cues separate real murmurs from normals")

    # The S1/S2 detector feeds the heart rate and the systolic/diastolic call,
    # so it needs measuring against hand-marked truth rather than asserting.
    seg = validate_segmentation()
    if seg:
        assert seg["recall"] > 0.85, f"S1/S2 recall dropped to {seg['recall']:.1%}"
        assert seg["precision"] > 0.80, f"S1/S2 precision dropped to {seg['precision']:.1%}"
        print("ok  S1/S2 detection holds against hand-marked ground truth")

    # Systole must come out shorter than diastole on a resting heart -- the whole
    # cycle labelling rests on that asymmetry, so verify it end to end.
    tfile = PASCAL / "set_a_timing.csv"
    if tfile.exists():
        ref = pd.read_csv(tfile)
        fname = ref.fname.iloc[0]
        y = sosfiltfilt(SOS, load_audio(PASCAL / fname))
        y = np.ascontiguousarray(y / (np.abs(y).max() + 1e-9), dtype=np.float32)
        times, _ = detect_sounds(y)
        _, systole, diastole = segment_cycle(times)
        assert systole and diastole, "no complete cycles found in a clean recording"
        s = np.median([b - a for a, b in systole])
        d = np.median([b - a for a, b in diastole])
        assert s < d, f"systole {s:.3f}s not shorter than diastole {d:.3f}s"
        print(f"ok  cycle labelling: systole {s:.3f}s < diastole {d:.3f}s")
        assert cycles_are_plausible(systole, diastole, 105)[0], \
            "gate rejected a clean recording"

    # The gate is what stops a failed segmentation from producing a clinical
    # sentence, so check it refuses nonsense as well as accepting good input.
    good = [(0.0, 0.25), (0.8, 1.05), (1.6, 1.85)]
    for bad, label in (
            ([(0.0, 0.25)], "single cycle"),
            (good, "impossible heart rate"),
            ([(0.0, 0.9), (1.8, 2.7)], "systole far too long"),
    ):
        spans, bpm = (bad, 300) if label == "impossible heart rate" else (bad, 90)
        dia = [(0.25, 0.8), (1.05, 1.6)] if label != "single cycle" else []
        assert not cycles_are_plausible(spans, dia, bpm)[0], f"gate let through: {label}"
    # An irregular rhythm must be refused even when every span looks individually fine.
    assert not cycles_are_plausible(
        good, [(0.25, 0.8), (1.05, 3.4)], 90)[0], "gate let through irregular cycles"
    print("ok  timing gate refuses implausible segmentation")

    # A saved model and this code must agree on the feature contract. Edit
    # features() without retraining and every prediction silently shifts, with
    # no error to notice -- so check the bundle rather than trust it.
    if MODEL.exists():
        b = joblib.load(MODEL)
        assert b["n_features"] == N_FEATURES, (
            f"{MODEL.name} expects {b['n_features']} features, code produces "
            f"{N_FEATURES} -- retrain with --train")
        assert b["sr"] == SR, f"{MODEL.name} trained at {b['sr']} Hz, code uses {SR}"
        assert 0.0 < b["threshold"] < 1.0, "saved threshold outside (0,1)"
        assert b["model"].n_features_in_ == N_FEATURES
        print(f"ok  saved model matches the current feature contract "
              f"({N_FEATURES} features, {SR} Hz)")

    print("all checks passed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="circor",
                    help="circor (default) and/or PASCAL a,b, comma-separated")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--recall", type=float, default=0.90,
                    help="recall target for the text report's screening row")
    ap.add_argument("--patients", type=int, default=1,
                    help="example patients per outcome on the review page")
    ap.add_argument("--skip-fetch", action="store_true", help="assume data is already present")
    ap.add_argument("--skip-audio", action="store_true",
                    help="skip review.html (it is the only stage needing ffmpeg)")
    ap.add_argument("--check", action="store_true", help="run self-checks and exit")
    ap.add_argument("--train", action="store_true",
                    help="fit a final model on all data and save it for deployment")
    ap.add_argument("--predict", nargs="+", metavar="WAV",
                    help="score new recordings with the saved model, then exit")
    ap.add_argument("--analyse", "--analyze", nargs="+", metavar="WAV", dest="analyse",
                    help="step-by-step analysis of a recording: quality, heart rate, "
                         "cycle segmentation, murmur timing, then the verdict")
    ap.add_argument("--transfer", action="store_true",
                    help="train on CirCor, test on the held-out CinC 2016 corpus "
                         "(adults, noisier) -- the generalisation test")
    ap.add_argument("--fetch-cinc", action="store_true",
                    help="download the CinC 2016 training set (181 MB)")
    args = ap.parse_args(argv)

    sets = tuple(s.strip() for s in args.sets.split(","))

    # Analysing and predicting need no dataset -- only the saved model. Check
    # them first so a deployed jacket never triggers a 449 MB download.
    if args.analyse:
        for p in args.analyse:
            analyse(p)
        return 0
    if args.predict:
        predict(args.predict)
        return 0

    if args.fetch_cinc:
        fetch_cinc()
        return 0
    if args.transfer:
        if not args.skip_fetch:
            fetch_circor()
            fetch_cinc()
        transfer_test(train_sets=sets, folds=args.folds)
        return 0

    if not args.skip_fetch and "circor" in sets:
        fetch_circor()
    if args.check:
        self_check()
        return 0
    if args.train:
        train_model(load_index(sets=sets), args.folds)
        return 0

    df = load_index(sets=sets)
    print(f"\n{len(df)} recordings, {df.patient.nunique()} patients, "
          f"{(df.label == 'murmur').sum()} murmurs")
    wprob, owner, prob = cross_val_scores(df, args.folds)
    print(f"{len(wprob)} windows x {N_FEATURES} features")

    truth = (df.label == "murmur").to_numpy().astype(int)
    thr = balanced_threshold(truth, prob)
    OUT.mkdir(exist_ok=True)

    text_report(df, prob, args.recall)
    export_detections(df, prob, thr, OUT / "detections.csv")

    cohort_chart(df, prob, args.recall, OUT / "cohort.png")
    print(f"wrote {OUT / 'cohort.png'}")

    d = df.assign(p=prob, t=truth)
    pat = d.groupby("patient").agg(p=("p", "max"), t=("t", "max"))
    pools = {
        "caught": pat[(pat.t == 1) & (pat.p > thr)].sort_values("p", ascending=False),
        "missed": pat[(pat.t == 1) & (pat.p <= thr)].sort_values("p"),
        "falsealarm": pat[(pat.t == 0) & (pat.p > thr)].sort_values("p", ascending=False),
        "clear": pat[(pat.t == 0) & (pat.p <= thr)].sort_values("p"),
    }
    for kind, g in pools.items():
        for pid in g.index[:args.patients]:
            out = OUT / f"patient_{kind}_{pid}.png"
            patient_chart(df, wprob, owner, pid, thr, out)
            print(f"wrote {out}")

    if args.skip_audio:
        print("\nskipped review.html (--skip-audio)")
        return 0
    if not shutil.which("ffmpeg"):
        print("\nffmpeg not on PATH -- skipping review.html. Everything else is written.")
        return 0

    print("\nbuilding audio cards ...")
    groups = build_cards(df, wprob, owner, prob, thr, args.patients)
    render_page(groups, thr, _plain_numbers(df, prob, thr), _operating_table(df, prob),
                OUT / "cohort.png", OUT / "review.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
