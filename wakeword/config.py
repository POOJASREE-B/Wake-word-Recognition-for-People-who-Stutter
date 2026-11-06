"""Central configuration. Every magic number of the project lives here."""
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RAW_DIR = ROOT / "raw"          # put the uploaded .zip files here
FRONTEND_DIR = ROOT / "frontend"

# ----------------------------------------------------------------- audio / features
SR = 16000                      # sample rate used everywhere
WIN = 400                       # 25 ms analysis window
HOP = 160                       # 10 ms hop  -> 100 frames / second
N_FFT = 512
N_MELS = 32
FMIN, FMAX = 60.0, 7600.0
CONTEXT = (-22, -15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15, 22)   # frame offsets stacked for the frame classifier

# ----------------------------------------------------------------- frame classes
CLASSES = ["bg", "hey", "goo", "gle", "other"]
BG, HEY, GOO, GLE, OTHER = range(5)
IGNORE = 5                      # frames excluded from frame-model training (look-alike words in negative clips)
KEYWORD_TEXT = "Hey Google"

# ----------------------------------------------------------------- detector
MIN_FRAMES = {HEY: 4, GOO: 4, GLE: 3}     # minimum duration of each keyword unit (10 ms frames)
MAX_GAP_FRAMES = 300                      # a block inside the phrase may last up to 3 s
CLIP_SECONDS = 6.0                        # longest audio analysed in a single call

# ----------------------------------------------------------------- decision rules (v1.2 fixes)
# Every keyword unit must be clearly HEARD somewhere on the decoded path: the peak frame posterior of Hey, of Goo
# and of gle must each reach UNIT_GATE.  Without this the grammar can force a stuttered 'H-h-hey' through a few weak
# 'goo'/'gle' frames and the verifier then accepts it ("it doesn't wait for Google").
UNIT_GATE = 0.30
# A filled pause ('Hey, um, Google') is allowed inside the phrase: gap states may explain OTHER frames, at this
# extra log-cost per frame so that a long unrelated word inside the phrase stays expensive.
FILLER_FRAME_PENALTY = -0.25
