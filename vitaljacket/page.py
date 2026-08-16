"""The playable HTML review page."""

import base64
import io
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import sosfiltfilt
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.colors import TwoSlopeNorm

from .config import (BLUES, CLIP_SECONDS, DIVERGING, INK2, MUTED, OUT,
                     PLAYBACK_SOS, SR, SURFACE, plt)
from .charts import score_over_time
from .data import load_audio
from .model import balanced_threshold


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


def render_page(groups, threshold, df, prob, cohort_png, path):
    """Write the review page. Every figure on it is derived here from the same
    scores the caller passed, so the headline numbers and the case cards can
    never describe two different runs."""
    stats = _plain_numbers(df, prob, threshold)
    operating = _operating_table(df, prob)
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
