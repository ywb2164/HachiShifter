"""Synthetic sanity checks for pitchfit's splitting rule (no HachiShifter)."""
import numpy as np
from pitchfit import fit_note


def contour(dur, wobble=0.0, slope=0.0, vib=0.0):
    t = np.arange(0, dur, 0.005)
    slow = slope * t / max(dur, 1e-9) * 100 + wobble * np.sin(2 * np.pi * 1.3 * t)
    v = vib * np.sin(2 * np.pi * 5.5 * t)
    return [{"time_seconds": float(x), "relative_cents": float(a + b), "without_vibrato_cents": float(a), "voiced": True} for x, a, b in zip(t, slow, v)]


def line(dur, f):
    t = np.arange(0, dur + 1e-9, 0.005)
    return [(float(x), float(f(x))) for x in t]


cases = {
    "short stable note (0.18 s)": (0.18, contour(0.18, wobble=15), line(0.18, lambda x: 64.0)),
    "short note, small slide (0.18 s)": (0.18, contour(0.18), line(0.18, lambda x: 64 + 0.3 * x / 0.18)),
    "long note, 6-step melisma (1.2 s)": (1.2, contour(1.2, wobble=20), line(1.2, lambda x: [64, 66, 67, 66, 64, 62][min(5, int(x / 0.2))])),
    "long note, vibrato only (1.2 s, ±50 c)": (1.2, contour(1.2, wobble=10), line(1.2, lambda x: 64 + 0.5 * np.sin(2 * np.pi * 5.8 * x))),
    "long note, slow 3-semitone glide (1.0 s)": (1.0, contour(1.0), line(1.0, lambda x: 64 + 3 * x)),
    "long note, glide + run (1.4 s)": (1.4, contour(1.4, wobble=15), line(1.4, lambda x: 64 + 2 * min(x, 0.3) / 0.3 if x < 0.6 else [67, 69, 67, 66, 64][min(4, int((x - 0.6) / 0.16))])),
}
for name, (dur, c, R) in cases.items():
    p = fit_note(c, dur, R)
    print(f"{name:42s} splits {len(p['splits']):2d}  d={p['drift']} m={p['modulation']}  err {p['err_unsplit_cents']:6.1f} -> {p['err_cents']:5.1f} c"
          + (f"  cuts at {[s[0] for s in p['splits']]}" if p['splits'] else ""))
