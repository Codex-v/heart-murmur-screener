"""Command line entry point. Every stage is reachable from here and nowhere else."""

import argparse
import shutil
import sys

import numpy as np

from .analysis import analyse
from .charts import cohort_chart, patient_chart
from .checks import self_check
from .config import MODEL, N_FEATURES, OUT
from .data import fetch_cinc, fetch_circor, load_index
from .model import (balanced_threshold, cross_val_scores, predict, train_model,
                    transfer_test)
from .page import build_cards, render_page
from .report import export_detections, text_report

DESCRIPTION = """\
Heart murmur screener -- the whole pipeline.

Detects VALVE blockage (stenosis) via the murmur it produces. Does NOT detect
coronary artery blockage, which produces no murmur, nor "heart block" in the
cardiology sense, which is an ECG finding.
"""


def build_parser():
    ap = argparse.ArgumentParser(
        prog="vital_jacket.py", description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="circor",
                    help="circor (default) and/or PASCAL a,b, comma-separated")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--recall", type=float, default=0.90,
                    help="recall target for the text report's screening row")
    ap.add_argument("--patients", type=int, default=1,
                    help="example patients per outcome on the review page")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="assume the data is already present")
    ap.add_argument("--skip-audio", action="store_true",
                    help="skip review.html (the only stage needing ffmpeg)")
    ap.add_argument("--check", action="store_true", help="run self-checks and exit")
    ap.add_argument("--train", action="store_true",
                    help="fit a final model on all data and save it for deployment")
    ap.add_argument("--predict", nargs="+", metavar="WAV",
                    help="score new recordings with the saved model, then exit")
    ap.add_argument("--analyse", "--analyze", nargs="+", metavar="WAV", dest="analyse",
                    help="step-by-step analysis: quality, heart rate, cycle "
                         "segmentation, murmur timing, then the verdict")
    ap.add_argument("--transfer", action="store_true",
                    help="train on CirCor, score corpora it has never seen")
    ap.add_argument("--fetch-cinc", action="store_true",
                    help="download the CinC 2016 training set (181 MB)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    sets = tuple(s.strip() for s in args.sets.split(","))

    # Analysing and predicting need no dataset -- only the saved model. Handle
    # them first so a deployed device never triggers a 449 MB download.
    if args.analyse:
        for p in args.analyse:
            analyse(p)
        return 0
    if args.predict:
        predict(args.predict)
        return 0

    if args.fetch_cinc:
        fetch_cinc()
        return 0
    if args.transfer:
        if not args.skip_fetch:
            fetch_circor()
            fetch_cinc()
        transfer_test(train_sets=sets, folds=args.folds)
        return 0

    if not args.skip_fetch and "circor" in sets:
        fetch_circor()
    if args.check:
        self_check()
        return 0
    if args.train:
        train_model(load_index(sets=sets), args.folds)
        return 0

    return run_reports(sets, args)


def run_reports(sets, args):
    """The default run: evaluate, then write every report from one set of scores."""
    df = load_index(sets=sets)
    print(f"\n{len(df)} recordings, {df.patient.nunique()} patients, "
          f"{(df.label == 'murmur').sum()} murmurs")
    wprob, owner, prob = cross_val_scores(df, args.folds)
    print(f"{len(wprob)} windows x {N_FEATURES} features")

    truth = (df.label == "murmur").to_numpy().astype(int)
    thr = balanced_threshold(truth, prob)
    OUT.mkdir(exist_ok=True)

    text_report(df, prob, args.recall)
    export_detections(df, prob, thr, OUT / "detections.csv")
    cohort_chart(df, prob, args.recall, OUT / "cohort.png")
    print(f"wrote {OUT / 'cohort.png'}")

    d = df.assign(p=prob, t=truth)
    pat = d.groupby("patient").agg(p=("p", "max"), t=("t", "max"))
    pools = {
        "caught": pat[(pat.t == 1) & (pat.p > thr)].sort_values("p", ascending=False),
        "missed": pat[(pat.t == 1) & (pat.p <= thr)].sort_values("p"),
        "falsealarm": pat[(pat.t == 0) & (pat.p > thr)].sort_values("p", ascending=False),
        "clear": pat[(pat.t == 0) & (pat.p <= thr)].sort_values("p"),
    }
    for kind, g in pools.items():
        for pid in g.index[:args.patients]:
            out = OUT / f"patient_{kind}_{pid}.png"
            patient_chart(df, wprob, owner, pid, thr, out)
            print(f"wrote {out}")

    if args.skip_audio:
        print("\nskipped review.html (--skip-audio)")
        return 0
    if not shutil.which("ffmpeg"):
        print("\nffmpeg not on PATH -- skipping review.html. Everything else is written.")
        return 0

    print("\nbuilding audio cards ...")
    groups = build_cards(df, wprob, owner, prob, thr, args.patients)
    render_page(groups, thr, df, prob, OUT / "cohort.png", OUT / "review.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
