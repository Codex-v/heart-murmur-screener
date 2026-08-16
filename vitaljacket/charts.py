"""Validation and per-patient review charts."""

import librosa
import numpy as np
from scipy.signal import sosfiltfilt

from .config import (AXIS, BLUE, BLUES, CRITICAL, DIVERGING, GOOD, INK, INK2,
                     MUTED, ORANGE, SOS, SR, SURFACE, plt)
from .data import load_audio
from .model import screening_threshold

from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


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
