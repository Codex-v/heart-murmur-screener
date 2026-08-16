"""Cross-validation, thresholds, calibration, training, inference."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict

from .config import CACHE_DIR, MODEL, N_FEATURES, SR
from .data import load_audio, load_index
from .features import build_matrix, features, windows


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


def _ece(y, p, bins=10):
    """Expected calibration error: mean gap between promise and outcome.

    Quantile bins, not equal-width -- most scores cluster low, so equal-width
    bins leave the top half nearly empty and the number becomes noise.
    """
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    err = tot = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            err += m.sum() * abs(p[m].mean() - y[m].mean())
            tot += m.sum()
    return err / max(tot, 1)


def fit_calibrator(prob, y, groups, kind="isotonic", folds=5):
    """Map raw scores onto observed murmur rates.

    The model is trained with class_weight='balanced', which tells it to behave
    as though murmurs were half the population. They are 16.5%. So its scores
    are systematically high -- 0.40 corresponds to roughly a 20% real rate --
    and quoting one as a probability to a clinician would overstate the risk
    twofold. This learns the correction from the out-of-fold scores.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    def new():
        if kind == "isotonic":
            return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        return LogisticRegression(C=1e6)

    def fit(model, p, t):
        if kind == "isotonic":
            model.fit(p, t)
        else:
            model.fit(_logit(p), t)
        return model

    def apply(model, p):
        return (model.predict(p) if kind == "isotonic"
                else model.predict_proba(_logit(p))[:, 1])

    # Honest evaluation: the calibrator never sees the scores it is scored on.
    # Fitting and evaluating on the same out-of-fold predictions would make any
    # calibrator look perfect, isotonic especially -- it can memorise the curve.
    out = np.zeros_like(prob)
    for tr, te in GroupKFold(n_splits=folds).split(prob, y, groups):
        m = fit(new(), prob[tr], y[tr])
        out[te] = apply(m, prob[te])
    final = fit(new(), prob, y)   # deployment calibrator, fitted on everything
    return final, out, apply


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibration_report(df, folds=5):
    """Compare calibrators and report what actually improves. Returns the best."""
    from sklearn.metrics import brier_score_loss

    _, _, prob = cross_val_scores(df, folds)
    y = (df.label == "murmur").to_numpy().astype(int)
    groups = df.patient.to_numpy()

    print(f"raw scores: mean {prob.mean():.3f} vs true murmur rate {y.mean():.3f}"
          f"  -> overstated by {prob.mean() / max(y.mean(), 1e-9):.1f}x")
    print(f"\n  {'method':<12}{'ECE':>9}{'Brier':>9}{'AUC':>9}   (ECE and Brier lower is better)")
    rows = [("raw", prob)]
    fitted = {}
    for kind in ("isotonic", "sigmoid"):
        final, oof, apply = fit_calibrator(prob, y, groups, kind, folds)
        fitted[kind] = (final, apply)
        rows.append((kind, oof))
    for name, p in rows:
        print(f"  {name:<12}{_ece(y, p):>9.3f}{brier_score_loss(y, p):>9.3f}"
              f"{roc_auc_score(y, p):>9.3f}")

    # Pick the best calibration that does NOT cost ranking. Isotonic is a step
    # function: it fits the curve tighter but collapses distinct scores into
    # ties, which loses AUC (~0.011 here) -- and AUC is what the screening
    # threshold spends. Sigmoid is strictly monotonic, so it reprices every
    # score without reordering any of them. Beyond about 0.03, both are accurate
    # enough that the difference cannot change a referral.
    scores = dict(rows)
    auc_raw = roc_auc_score(y, prob)
    ok = [k for k in ("sigmoid", "isotonic")
          if auc_raw - roc_auc_score(y, scores[k]) <= 0.005]
    best = min(ok or ["sigmoid"], key=lambda k: _ece(y, scores[k]))
    print(f"\n  chosen: {best}  (best calibration that keeps AUC within 0.005)")
    print(f"  ECE {_ece(y, prob):.3f} -> {_ece(y, scores[best]):.3f}, "
          f"a {_ece(y, prob) / max(_ece(y, scores[best]), 1e-9):.1f}x improvement")
    return best, fitted[best]


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

    kind, (calib, _) = calibration_report(df, folds)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "threshold": thr, "n_features": N_FEATURES, "sr": SR,
                 "calibrator": calib, "calibrator_kind": kind,
                 "prevalence": float(y.mean()),
                 "trained_on": f"{len(df)} recordings, {df.patient.nunique()} patients",
                 "auc_crossval": float(roc_auc_score(y, prob))}, path)
    print(f"\nwrote {path}\n  threshold {thr:.3f} | cross-validated AUC "
          f"{roc_auc_score(y, prob):.3f} | trained on {len(df)} recordings")
    return path


def apply_calibration(bundle, raw):
    """Raw score -> probability, using whichever calibrator the bundle carries.

    Returns None for a model saved before calibration existed, so callers fall
    back to showing the raw score rather than inventing a probability.
    """
    calib, kind = bundle.get("calibrator"), bundle.get("calibrator_kind")
    if calib is None:
        return None
    raw = np.atleast_1d(np.asarray(raw, dtype=float))
    return (calib.predict(raw) if kind == "isotonic"
            else calib.predict_proba(_logit(raw))[:, 1])


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
        # The threshold is applied to the RAW score, never the calibrated one.
        # Calibration only reprices; the operating point was tuned on raw scores
        # and re-deriving it from probabilities would move the decision.
        cal = apply_calibration(bundle, score)
        shown = (f"{cal[0]:>4.0%} chance  raw {score:.3f}" if cal is not None
                 else f"     raw {score:.3f}      ")
        print(f"  {p.name:<34}{shown}   "
              f"{'MURMUR DETECTED' if flag else 'clear':<16} ({len(ws)} windows)")
        results.append({"file": str(p), "score": score, "flag": bool(flag),
                        "probability": None if cal is None else float(cal[0])})
    if bundle.get("calibrator") is not None:
        print(f"\nProbabilities are calibrated against a {bundle['prevalence']:.0%} "
              "murmur rate.\nIn a population where murmurs are rarer or commoner, "
              "they shift accordingly.")
    print("\nA flag means: refer for an echocardiogram. It is not a diagnosis,\n"
          "and it says nothing about coronary artery disease.")
    return results
