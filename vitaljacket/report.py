"""The text report and the detections CSV."""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from .model import cross_val_scores


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
