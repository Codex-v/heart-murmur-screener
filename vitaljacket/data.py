"""Dataset download, labelling and audio loading."""

import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import librosa
import pandas as pd

from .config import CINC, CINC_URL, CIRCOR, CIRCOR_TOP, CIRCOR_URL, PASCAL, SR


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
