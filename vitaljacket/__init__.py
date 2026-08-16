"""Heart murmur screener.

Detects VALVE blockage (stenosis) through the murmur turbulent flow produces.
Does NOT detect coronary artery blockage -- that needs angiography-labelled
training data, which no open dataset provides -- nor "heart block" in the
cardiology sense, which is an ECG finding rather than a sound.

Module map, in pipeline order:

    config      paths, constants, filter coefficients, chart palette
    data        dataset download, per-site labelling, audio loading
    features    4 s windowing and the 149-number feature vector
    model       cross-validation, thresholds, calibration, training, inference
    analysis    beats, cycle segmentation, murmur timing, signal quality
    report      the text report and the detections CSV
    charts      validation and per-patient review figures
    page        the playable HTML review page
    checks      self-checks for every score-inflating invariant
    cli         argument parsing and stage dispatch

Not a medical device. Not validated, not certified, not for clinical use.
"""

__all__ = ["main"]


def main(argv=None):
    from .cli import main as _main
    return _main(argv)
