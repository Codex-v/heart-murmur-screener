"""Paths, constants, filters and the chart palette. Imported by everything."""

from pathlib import Path

import librosa
import matplotlib
from scipy.signal import butter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


ROOT_DIR = Path(__file__).resolve().parent.parent   # the repo root, not the package
HEART = ROOT_DIR / "Heart Sound"
PASCAL = HEART / "dataset"
CIRCOR = HEART / "circor"
CINC = HEART / "cinc2016"
CACHE_DIR = HEART / ".cache"
OUT = ROOT_DIR / "reports"
MODEL = ROOT_DIR / "murmur_model.joblib"

SR = 4000              # CirCor and PASCAL set_b are native 4 kHz; set_a resamples down
N_MELS, N_FFT, HOP = 40, 512, 128
WIN, STEP = 4 * SR, 2 * SR   # 4 s analysis windows, 50% overlap
N_FEATURES = 149
CLIP_SECONDS = 14      # audio excerpt on the review page

CIRCOR_URL = "https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/"
CIRCOR_TOP = "the-circor-digiscope-phonocardiogram-dataset-1.0.3/"
CINC_URL = "https://physionet.org/files/challenge-2016/1.0.0/training.zip"

# 25-900 Hz for analysis: below is body/handling rumble, above is past anything
# diagnostic at this rate. Murmurs live around 100-600 Hz.
SOS = butter(4, (25, 900), btype="bandpass", fs=SR, output="sos")
# Wider for playback -- the band an electronic stethoscope passes.
PLAYBACK_SOS = butter(4, (20, 1000), btype="bandpass", fs=SR, output="sos")
MEL_HZ = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=1000)

# Validated light-mode data-viz palette. The spectrogram ramp is deliberately
# SINGLE HUE: a viridis/jet rainbow on a magnitude scale creates banding that
# reads as structure in the signal, which on a diagnostic image is a correctness
# problem, not a cosmetic one.
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, CRITICAL, GOOD = "#2a78d6", "#eb6834", "#d03b3b", "#0ca30c"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("seq_blue", SEQ)
DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", CRITICAL])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",  # never dashed
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 130,
})
