# Heart murmur screener

Acoustic screening for heart murmurs from phonocardiogram recordings. Listens to each
chest probe and scores how likely it is to contain a murmur — the turbulent-flow sound a
narrowed or leaking heart valve makes.

The whole pipeline is one file: `vital_jacket.py`.

```bash
pip install -r requirements.txt
python vital_jacket.py            # download data, train, evaluate, write every report
python vital_jacket.py --check    # self-checks only
```

Outputs land in `reports/`: validation charts, per-patient review pages, a CSV of every
recording ranked by score, and a self-contained HTML page with playable audio.

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

## Limitations

- **The training population is paediatric** (664 children, 126 infants, 72 adolescents,
  6 neonates), so its murmurs are largely innocent or congenital. Performance on adults,
  whose murmurs are typically degenerative aortic stenosis, is unmeasured.
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
