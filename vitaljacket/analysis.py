"""What is actually in a recording: beats, cycles, timing, quality."""

from pathlib import Path

import joblib
import librosa
import numpy as np
from scipy.signal import sosfiltfilt

from .config import MEL_HZ, MODEL, SOS, SR
from .data import load_audio
from .features import features, windows
from .model import apply_calibration


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
        cal = apply_calibration(model, score)
        print(f"    raw score    {score:.3f}   (threshold {thr:.3f})")
        if cal is not None:
            print(f"    probability  {cal[0]:.0%}   of a murmur being present, "
                  f"calibrated against a {model['prevalence']:.0%} rate")
        print(f"    per window   " + " ".join(f"{s:.2f}" for s in scores[:12]))
        print(f"    VERDICT      {'MURMUR DETECTED' if score > thr else 'clear'}")
        if q["verdict"] == "poor":
            print("    Poor signal quality -- this verdict is unreliable.")
        print("\n    A flag means: refer for an echocardiogram. Not a diagnosis, and\n"
              "    it says nothing about coronary artery disease.")
    return {"duration": dur, "quality": q, "bpm": bpm, "cycles": len(systole),
            "timing": mt}
