"""Check pitchfit's predicted error against HachiShifter's rendered target pitch on real notes."""
import json, sys
from pathlib import Path
import numpy as np
import tools
from tools import Session
from pitchfit import fit_note

RUN = Path("runs/t10_wonder_multi_rules_v1_1107-150441")
task = json.loads(Path("tasks_local/t10_wonder_multi.json").read_text(encoding="utf-8"))
tools.RENDER.clear(); tools.RENDER.update({"pitch_algorithm": "mld5", "stretch_algorithm": "melodyne-hybrid", "render_order": "process-then-splice", "merge_contiguous_source": True})
tools.FIT.clear(); tools.FIT.update(vowel_fit="trim", max_vowel_compression=1.5)
out = RUN / "replay" / "pitchfit_check"; out.mkdir(parents=True, exist_ok=True)
s = Session(task, out)
s.apply_picks(json.loads((RUN / "summary.json").read_text(encoding="utf-8"))["picks"])
curves = s._curves("reference"); tune = s._tune()


def snap_notes():
    tr = next(t for t in s.mcp.call_json("project_snapshot")["tracks"] if t["id"] == s.track_id)
    return {n["id"]: n for c in tr["clips"] for n in c["notes"]}


def R_for(n):
    pts = curves.get(str(n["index"])) or []
    pts = [(float(t), float(v)) for t, v in pts if 0 <= float(t) <= n["duration"]]
    return pts if len(pts) >= 2 else [(0.0, n["midi"] + tune), (n["duration"], n["midi"] + tune)]


def rendered_err(note, R):
    rt = np.array([p[0] for p in R]); rv = np.array([p[1] for p in R])
    errs = [100 * (note["midi"] + p["rendered_target_cents"] / 100 - np.interp(p["time_seconds"], rt, rv))
            for p in note["contour"] if p["voiced"] and p["time_seconds"] <= note["duration_seconds"]]
    return float(np.mean(np.abs(errs))) if errs else None


sn = snap_notes()
idx = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [5, 12, 20, 33, 47, 60, 75, 90]
for i in idx:
    n = s.notes[i]; R = R_for(n); note = sn[n["id"]]
    plan = fit_note(note["contour"], note["duration_seconds"], R)
    before = rendered_err(note, R)
    s.mcp.call("edit_notes_pitch", note_id=n["id"], action="set", midi=plan["center_midi"])
    s.mcp.call("set_note", note_id=n["id"], drift=plan["drift"], modulation=plan["modulation"])
    after = rendered_err(snap_notes()[n["id"]], R)
    print(f"note {i:3d} dur {note['duration_seconds']:.3f} d={plan['drift']} m={plan['modulation']} center {plan['center_midi']:.2f} "
          f"(score {n['midi']}) predicted {plan['err_unsplit_cents']}c rendered-before(default) {before and round(before,1)}c after {after and round(after,1)}c splits {plan['splits']}")
# split test on the first note that wants a split
for i in range(len(s.notes)):
    n = s.notes[i]
    if not n.get("pick"): continue
    note = snap_notes()[n["id"]]; R = R_for(n); plan = fit_note(note["contour"], note["duration_seconds"], R)
    if plan["splits"]:
        t_cut, c2 = plan["splits"][0]
        r = s.mcp.call_json("split_note", note_id=n["id"], local_seconds=t_cut)
        print("split note", i, "at", t_cut, "->", r)
        after = snap_notes()
        a, b = after.get(r.get("note_id")), after.get(r.get("new_note_id"))
        for tag, x in (("A", a), ("B", b)):
            if x: print(f"  piece {tag}: start {x['start_seconds']:.3f} dur {x['duration_seconds']:.3f} midi {x['midi']:.2f} src_center {x['source_midi_center']:.2f} drift {x['drift']} mod {x['modulation']} contour {len(x['contour'])} material {bool(x.get('native_segments'))}")
        break
s.close()
