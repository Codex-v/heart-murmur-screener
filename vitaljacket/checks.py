"""Self-checks. Every invariant that would silently inflate a score."""

import joblib
import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt
from sklearn.model_selection import GroupKFold

from .config import CIRCOR, MODEL, N_FEATURES, PASCAL, SOS, SR, WIN
from .analysis import (cycles_are_plausible, detect_sounds, segment_cycle,
                       shannon_envelope)
from .data import load_audio, load_index, patient_and_position
from .features import features, windows


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
