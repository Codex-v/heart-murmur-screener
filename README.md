# Heart murmur screener

Acoustic screening for heart murmurs from phonocardiogram recordings. Listens to each
chest probe and scores how likely it is to contain a murmur — the turbulent-flow sound a
narrowed or leaking heart valve makes.

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

  84693_TV.wav    100% chance  raw 0.999   MURMUR DETECTED  (9 windows)
  50164_MV.wav      1% chance  raw 0.028   clear            (14 windows)
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
    raw score    0.999   (threshold 0.364)
    probability  100%    of a murmur being present
    VERDICT      MURMUR DETECTED
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

## Layout

`vital_jacket.py` is a thin entry point; the pipeline lives in `vitaljacket/`, one
module per stage:

| Module | Lines | What it owns |
|---|---:|---|
| `config.py` | 61 | Paths, constants, filter coefficients, chart palette |
| `data.py` | 193 | Dataset download, per-site labelling, audio loading |
| `features.py` | 87 | 4 s windowing, the 149-number feature vector, caching |
| `model.py` | 322 | Cross-validation, thresholds, calibration, training, inference |
| `analysis.py` | 298 | Beat detection, cycle segmentation, murmur timing, signal quality |
| `report.py` | 66 | Text report and the detections CSV |
| `charts.py` | 196 | Validation and per-patient review figures |
| `page.py` | 477 | The playable HTML review page |
| `checks.py` | 178 | Self-checks for every score-inflating invariant |
| `cli.py` | 139 | Argument parsing and stage dispatch |

Dependencies run one way: `config` → `data` → `features` → `model` → everything else.
Nothing imports `cli`.

Features are log-mel statistics, MFCCs with deltas, spectral shape descriptors, and five
hand-built murmur cues (brightness, duty cycle, non-peak brightness, dynamic range, and
high-frequency energy in the gaps between beats). The cue directions were measured on
labelled audio rather than assumed, and a self-check re-verifies them. The classifier is
gradient-boosted trees; deliberately not a CNN, since 489 positive recordings will overfit
one and this is a baseline to beat.

## Reading the score

The model is trained with `class_weight="balanced"`, which tells it to behave as though
murmurs were half the population. They are 16.5%. Its raw scores are therefore
systematically high, so `--train` also fits a calibrator and stores it in the model file:

```
84693_TV.wav    100% chance  raw 0.999   MURMUR DETECTED
84826_MV.wav     25% chance  raw 0.394   MURMUR DETECTED
50164_MV.wav      1% chance  raw 0.028   clear
```

That middle row is the reason it matters: a raw score just over the threshold is a
1-in-4 chance, not a near-certainty.

| | ECE | Brier | AUC |
|---|---|---|---|
| Raw scores | 0.094 | 0.098 | 0.822 |
| Platt / sigmoid | **0.028** | 0.087 | **0.821** |
| Isotonic | 0.014 | 0.086 | 0.811 |

Sigmoid is chosen over the better-calibrated isotonic because isotonic is a step function:
it collapses distinct scores into ties and costs 0.011 AUC, and AUC is what the screening
threshold spends. Both are accurate enough that the remaining difference cannot change a
referral. The calibrator is fitted and evaluated under separate patient-grouped folds —
fitting and scoring on the same predictions would make any calibrator look perfect.

**The threshold is still applied to the raw score, never the calibrated one.** The
operating point was tuned on raw scores; re-deriving it from probabilities would move the
decision. Calibration reprices, it does not re-decide.

**Probabilities carry the training population's prevalence.** They are calibrated against
a 16.5% murmur rate. Deployed where murmurs are rarer or commoner, they shift accordingly.

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
the numbers worth quoting.

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
