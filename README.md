# Heart murmur screener

Acoustic screening for heart murmurs from phonocardiogram recordings. Listens to each
chest probe and scores how likely it is to contain a murmur — the turbulent-flow sound a
narrowed or leaking heart valve makes.

The whole pipeline is one file: `vital_jacket.py`.

```bash
pip install -r requirements.txt

python vital_jacket.py                    # download data, train, evaluate, write reports
python vital_jacket.py --train            # fit a final model and save it for deployment
python vital_jacket.py --analyse a.wav    # step-by-step analysis of one recording
python vital_jacket.py --predict a.wav    # score new recordings with the saved model
python vital_jacket.py --transfer         # test on corpora it has never seen
python vital_jacket.py --check            # self-checks only
```

Outputs land in `reports/`: validation charts, per-patient review pages, a CSV of every
recording ranked by score, and a self-contained HTML page with playable audio.

### Scoring a new recording

```
$ python vital_jacket.py --predict recordings/*.wav
model: murmur_model.joblib | threshold 0.364 | 2964 recordings, 873 patients

  84693_TV.wav      score 0.999  MURMUR DETECTED  (loudest window 0.999, 9 windows)
  50164_MV.wav      score 0.028  clear            (loudest window 0.068, 14 windows)
```

`--predict` needs only the saved model, so a deployed device never touches the dataset.
The decision threshold is stored inside the model file: a score is meaningless without
the cut-off it was tuned against, and shipping them separately is how a screener
silently changes its mind between versions.

Any audio format `libsndfile` reads works, at any sample rate — input is resampled to
4 kHz internally.

### Analysing a recording

`--predict` gives a number. `--analyse` shows the working — signal quality, heart rate,
cycle segmentation, where the extra sound falls, the acoustic cues, then the verdict:

```
[2] Signal quality: USABLE
    beat-to-gap contrast  0.84   (>0.35 good, <0.20 poor)
    click artifacts       8.9%   (handling noise)

[3] Heart sounds detected: 66
    heart rate    100 bpm   (cycle 0.60 s)

[4] Cycle segmentation: 32 complete cycles
    systole  (S1->S2)  0.265 s
    diastole (S2->S1)  0.327 s

[5] Where the extra sound falls
    systolic skew  +0.14
    -> No clear phase preference (healthy hearts skew +0.14 here by default)

[7] Model verdict
    score  0.999  (threshold 0.364)   VERDICT  MURMUR DETECTED
```

The S1/S2 detector is measured, not assumed: against the PASCAL challenge's 390
hand-marked heart sounds it reaches **96.7% recall at 11 ms median timing error**, and
the self-check re-runs that comparison.

Two deliberate refusals in this output:

- **Timing is withheld when segmentation is unreliable.** If the detected rhythm is
  irregular or the rate implausible, section 5 prints nothing but the reason. A sentence
  naming candidate valve lesions, derived from a ten-second "diastole", reads as a
  clinical finding and is worse than silence. About 1 in 8 recordings is gated off.
- **Phase timing says *which*, never *whether*.** Calibrated on 108 labelled recordings,
  gap energy does not separate murmur from normal at all (healthy median 1.13, murmur
  0.99) — only the systolic-vs-diastolic skew differs, and weakly (+0.14 vs +0.24). So
  the phase call is reported as conditional on the model's verdict, with both reference
  values printed.

## Pipeline

| Stage | Where |
|---|---|
| Dataset download | `fetch_circor()` — 449 MB, idempotent |
| Labelling | `load_index()` — per auscultation site, patients grouped |
| Beat detection | `shannon_envelope()`, `detect_sounds()` — validated against ground truth |
| Cycle segmentation | `segment_cycle()` — S1/S2, systole vs diastole |
| Murmur timing | `murmur_timing()`, gated by `cycles_are_plausible()` |
| Signal quality | `signal_quality()` — contrast, clicks, clipping |
| Feature extraction | `windows()`, `features()` — 4 s windows, 149 features each |
| Training + evaluation | `cross_val_scores()` — patient-grouped 5-fold |
| Deployment model | `train_model()` — final fit, saved with its threshold |
| Inference | `predict()` — new recordings |
| Reports | `cohort_chart()`, `patient_chart()`, `render_page()` |

Features are log-mel statistics, MFCCs with deltas, spectral shape descriptors, and five
hand-built murmur cues (brightness, duty cycle, non-peak brightness, dynamic range, and
high-frequency energy in the gaps between beats). The cue directions were measured on
labelled audio rather than assumed, and a self-check re-verifies them. The classifier is
gradient-boosted trees; deliberately not a CNN, since 489 positive recordings will overfit
one and this is a baseline to beat.

## What it detects — and what it does not

**Detects: valve blockage (stenosis).** A narrowed or leaking valve forces blood through a
small opening; that turbulence is audible as a murmur, and it is what the model is
trained on.

**Does NOT detect coronary artery blockage** — the kind that causes heart attacks.
Training that would require recordings labelled against coronary angiography, which no
open dataset provides. It also does not detect *heart block* in the cardiology sense
(atrioventricular conduction block), which is an ECG finding, not a sound.

A murmur is a finding, not a diagnosis — many are innocent. The intended output is "this
chest sounds abnormal, send it for an echocardiogram".

## Performance

Five-fold cross-validation, **grouped by patient**, on CirCor DigiScope (2964 recordings,
873 patients, 489 with an audible murmur):

| | ROC-AUC |
|---|---|
| Per probe | 0.822 |
| Per patient | 0.808 |

At the balanced operating point, per patient: finds **56 of every 100** murmur cases and
correctly clears **95 of every 100** healthy ones.

Headline accuracy is 87%, but that figure flatters the tool — roughly 79% of patients are
healthy, so always answering "healthy" would already score 79%. Recall and specificity are
the numbers worth quoting. The model's raw score is deliberately class-weighted and is
**not** a calibrated probability; compare it to a threshold, do not read 0.40 as "40%".

## Does it generalise?

```bash
python vital_jacket.py --transfer     # train on CirCor, score two unseen corpora
```

A cross-validated score only says how the model does on *more of the same data* — same
device, same clinic, same population. For a wearable, the number that matters is how it
does somewhere else.

| Test | Target | ROC-AUC |
|---|---|---|
| CirCor, cross-validated | murmur audible at this site | **0.822** |
| → PASCAL set_b, unseen | murmur audible at this site | **0.799** |
| → CinC 2016, unseen | any confirmed cardiac diagnosis | 0.549 |

**The model transfers.** Against a corpus it has never seen, scored on the same question
it was trained to answer, performance drops by 0.023 — 58.9% recall at 92.5% specificity.

**The CinC score is not a generalisation failure.** That corpus labels "does this patient
have a confirmed cardiac diagnosis", and its abnormal group explicitly includes *coronary
artery disease*, which produces no murmur. Scoring near chance there is empirical
confirmation of the scope boundary at the top of this README: it hears valves, not
arteries. Quote it as evidence for that limit, never as this model's performance.

### A warning about pooled CinC 2016

Cross-validating on pooled CinC gives **ROC-AUC 0.973**. That number is an artifact, and
anyone reporting a figure of that magnitude on this corpus should check for it.

Its six sub-databases were collected on different equipment and carry very different
abnormal rates — 8.5% in `training-e`, 77.4% in `training-c`. A classifier can identify
which sub-database a recording came from **with 96.3% accuracy from the audio alone**, so
it can score well by recognising the recording device and reciting that device's base
rate, without ever detecting pathology. Leave-one-database-out across all seven sources
gives a median AUC of 0.565.

Never train or cross-validate on pooled CinC. It is used here only as a held-out test set.

## Limitations

- **The training population is paediatric** (664 children, 126 infants, 72 adolescents,
  6 neonates), so its murmurs are largely innocent or congenital. The transfer test above
  uses PASCAL set_b, which is also clinical rather than adult-ambulatory, so performance
  on adults remains unmeasured — no open corpus pairs adult recordings with per-site
  murmur labels.
- **Recordings are clinical, not ambulatory** — captured with a stethoscope held still. A
  body-worn probe will carry far more motion and friction artifact.
- Most murmurs in the data are faint (grade I–II of VI).

## Data

[The CirCor DigiScope Phonocardiogram Dataset v1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/),
Oliveira et al., PhysioNet — de-identified, open access, ODC-By 1.0. Downloaded by the
script; not vendored here. Attribution is required and is preserved in the generated
report page. The PASCAL Classifying Heart Sounds Challenge set is optionally supported
via `--sets a,b`.

## Disclaimer

**Not a medical device.** Not validated, not certified, and not for clinical use. Nothing
it produces is a diagnosis.
