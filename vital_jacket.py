#!/usr/bin/env python
"""Vital Jacket -- heart-sound murmur screener, entry point.

    python vital_jacket.py            # fetch data, train, evaluate, write every report
    python vital_jacket.py --help     # stage selection and options

Produces, in reports/:
    cohort.png          validation charts: ROC, score separation, confusion, calibration
    patient_*.png       per-probe review pages, one per clinical outcome
    detections.csv      every recording ranked by score, with its verdict
    review.html         playable clinical review page (needs ffmpeg)

WHAT THIS DETECTS
    Valve blockage (stenosis). A narrowed or leaking valve forces blood through a
    small opening; the turbulence is audible as a murmur, and that is what the
    model is trained on.

WHAT IT DOES NOT DETECT
    Coronary artery blockage -- the kind that causes heart attacks. Training that
    needs recordings labelled against coronary angiography, which no open dataset
    has. Nor does it detect "heart block" in the cardiology sense (AV conduction
    block), which is an ECG finding, not a sound.

    A murmur is a finding, not a diagnosis. The honest output is "this chest
    sounds abnormal, send it for an echocardiogram".

Data: CirCor DigiScope Phonocardiogram Dataset v1.0.3 (Oliveira et al., PhysioNet),
de-identified, ODC-By 1.0 -- redistributable with attribution. Optional: the PASCAL
Classifying Heart Sounds Challenge set, if present under Heart Sound/dataset/.

Not a medical device. Not validated, not certified, not for clinical use.
"""

import sys

from vitaljacket.cli import main

if __name__ == "__main__":
    sys.exit(main())
