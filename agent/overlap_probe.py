"""Does a note extended into its neighbour keep sounding (same track, same score clip), with/without native_audio_overlap?"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
import tools
from tools import Session
RUN = Path("runs/t10_wonder_multi_rules_v1_1107-150441")
task = json.loads(Path("tasks_local/t10_wonder_multi.json").read_text(encoding="utf-8"))
tools.RENDER.clear(); tools.RENDER.update({"pitch_algorithm": "mld5", "stretch_algorithm": "melodyne-hybrid", "render_order": "process-then-splice", "merge_contiguous_source": True})
tools.FIT.clear(); tools.FIT.update(vowel_fit="trim", max_vowel_compression=1.5)
A, B, EXT = 20, 21, 0.06
def render(overlap, keep_a, keep_b):
    out = RUN / "replay" / f"overlap_probe_{int(overlap)}{int(keep_a)}{int(keep_b)}"; out.mkdir(parents=True, exist_ok=True)
    s = Session(task, out)
    picks = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))["picks"]
    s.apply_picks({k: v for k, v in picks.items() if int(k) in (A, B)})
    for i, n in enumerate(s.notes):
        if i not in (A, B): s.mcp.call("remove_note", note_id=n["id"])
    if overlap: s.mcp.call("set_track", track_id=s.track_id, native_audio_overlap=True, smooth_overlaps=True)
    a, b = s.notes[A], s.notes[B]
    s.mcp.call("resize_note", note_id=a["id"], start_seconds=s._note_snapshot(a["id"])["start_seconds"], duration_seconds=a["duration"] + EXT)
    s.mcp.call("replace_note_material", roots=s.roots, choices=[{"note_id": a["id"], **{f: a["pick"][f] for f in ("audio_path", "audio_id", "region_id", "alias")}}],
               game_model_dir=str(tools.MODELS / "game" / "medium"), fcpe_model=str(tools.MODELS / "fcpe" / "fcpe.onnx"), **tools.FIT)
    if not keep_a: s.mcp.call("set_note", note_id=a["id"], gain=0.0)
    if not keep_b: s.mcp.call("set_note", note_id=b["id"], gain=0.0)
    wav = out / "r.wav"; s.mcp.call("export_wav", path=str(wav)); s.close()
    y, sr = sf.read(wav); y = y.mean(1) if y.ndim > 1 else y
    return y, sr, a, b

for overlap in (True,):
    ya, sr, a, b = render(overlap, True, False); yb, _, _, _ = render(overlap, False, True); yab, _, _, _ = render(overlap, True, True)
    n = min(len(ya), len(yb), len(yab)); end = a["start"] + a["duration"]
    pre = n / sr - (b["start"] + b["duration"]); pre = pre if 0 < pre < 0.6 else 0
    i0, i1 = int((end + pre) * sr), int((end + EXT + pre) * sr)
    rms = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
    print(f"overlap={overlap}: in the overlap window  A-alone {rms(ya[i0:i1]):.1f} dB  B-alone {rms(yb[i0:i1]):.1f} dB  both {rms(yab[i0:i1]):.1f} dB  "
          f"| both minus (A+B) {rms(yab[i0:i1] - ya[i0:i1] - yb[i0:i1]):.1f} dB  both minus B {rms(yab[i0:i1] - yb[i0:i1]):.1f} dB")
