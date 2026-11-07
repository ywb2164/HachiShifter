"""Rebuild the human teio project through the agent's score-material path and render it.

Same samples, same source spans, same note times, pitch centres, drift/modulation, formant and gains as
teio.mpd; only the path differs (HachiShifter's own source analysis + material binding instead of the
Melodyne import). The Melodyne-path render of teio is sample-identical to Melodyne's export, so per-note
differences here isolate host-side adaptation issues of the agent path.

    python teio_rebuild.py <teio_snap.json> <out_dir> [--dump DIR]
"""
import argparse, json, os, shutil, uuid
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools")); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import otomad_config as cfg  # noqa: E402
from mcp_client import HachiMcp  # noqa: E402

LIB = Path(__file__).resolve().parents[1] / "library" / "teio_ref"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("snap"); ap.add_argument("out"); ap.add_argument("--dump", default=None)
    ap.add_argument("--algo", default=None, help="override pitch algorithm (default: teio's track setting)")
    a = ap.parse_args()
    snap = json.load(open(a.snap, encoding="utf-8")); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    track = snap["tracks"][0]; bpm = snap["bpm"]
    clips = sorted(track["clips"], key=lambda c: c["start_seconds"])
    # 1. mirror sources under ASCII names (fresh each run, so the registrations are exactly ours)
    if LIB.exists(): shutil.rmtree(LIB)
    LIB.mkdir(parents=True)
    names = {}
    for c in clips:
        src = c["source_file"]
        if src not in names:
            names[src] = LIB / f"src{len(names):02d}.wav"; shutil.copy2(src, names[src])
    # 2. one region per note = its exact source span, P at the span start
    rows_by_file, notes = {}, []
    for i, c in enumerate(clips):
        n = c["notes"][0]
        f = names[c["source_file"]]
        aid = rows_by_file.setdefault(str(f), {"audio_id": "a" + uuid.uuid4().hex, "rows": []})["audio_id"]
        rid = "p" + uuid.uuid4().hex; alias = f"n{i:02d}"
        import soundfile as sf
        a0 = c["source_offset_seconds"]; a1 = min(a0 + c["source_duration_seconds"], sf.info(str(f)).duration - 0.0005)
        rows_by_file[str(f)]["rows"].append({"name": alias, "region_start_seconds": round(a0, 6), "region_end_seconds": round(a1, 6),
                                             "alignment_seconds": round(a0, 6), "fixed_duration_seconds": 0.0, "material_audio_id": aid,
                                             "material_region_id": rid, "native_provenance": "teio-rebuild", "native_confidence": 1.0})
        notes.append(dict(i=i, start=c["start_seconds"] + n["start_seconds"], dur=n["duration_seconds"], midi=n["midi"], drift=n["drift"],
                          modulation=n["modulation"], formant=n["formant_semitones"], gain=n["gain"] * c["gain"],
                          audio=str(f), audio_id=aid, region=rid, alias=alias))
    env = dict(os.environ)
    if a.dump: env["HACHI_MLD5_DUMP"] = a.dump
    os.environ.update(env)
    m = HachiMcp(cfg.EXE, [f"--roots={LIB}"])
    for f, d in rows_by_file.items():
        ok, t = m.call("sample_settings_save", audio_path=f, rows=sorted(d["rows"], key=lambda r: r["region_start_seconds"]))
        if not ok: raise RuntimeError("sample_settings_save " + f + ": " + t[:300])
    # 3. spread notes over tracks so no track holds overlapping notes
    lanes = []
    for nt in notes:
        for lane in lanes:
            if lane[-1]["start"] + lane[-1]["dur"] <= nt["start"] + 1e-6:
                lane.append(nt); break
        else:
            lanes.append([nt])
    settings = {k: track[k] for k in ("pitch_algorithm", "stretch_algorithm", "render_order", "merge_contiguous_source") if k in track}
    if a.algo: settings["pitch_algorithm"] = a.algo
    m.call("project_new"); m.call("set_tempo", bpm=bpm)
    for li, lane in enumerate(lanes):
        lines = ["[#VERSION]", "UST Version1.2", "[#SETTING]", f"Tempo={bpm:.2f}", "Tracks=1", f"ProjectName=teio_lane{li}", "VoiceDir=", "Mode2=True"]
        k = 0; pos = 0.0; tick = 480 * bpm / 60.0
        for nt in lane:
            gap = round((nt["start"] - pos) * tick)
            if gap > 0:
                lines += [f"[#{k:04d}]", f"Length={gap}", "Lyric=R", "NoteNum=60"]; k += 1
            ln = max(1, round(nt["dur"] * tick))
            lines += [f"[#{k:04d}]", f"Length={ln}", f"Lyric={nt['alias']}", f"NoteNum={round(nt['midi'])}", "Modulation=0"]; k += 1
            pos = (pos + gap / tick) + ln / tick
        lines.append("[#TRACKEND]")
        ust = out / f"lane{li}.ust"; ust.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
        ok, t = m.call("import_ust", path=str(ust))
        if not ok: raise RuntimeError("import_ust: " + t)
    snapshot = m.call_json("project_snapshot")
    tracks = snapshot["tracks"]
    assert len(tracks) == len(lanes), (len(tracks), len(lanes))
    choices = []
    for tr, lane in zip(tracks, lanes):
        m.call("set_track", track_id=tr["id"], **settings)
        sung = sorted((x for c in tr["clips"] for x in c["notes"] if (x.get("label") or "").upper() != "R"), key=lambda x: x["start_seconds"])
        assert len(sung) == len(lane), (len(sung), len(lane))
        for x, nt in zip(sung, lane):
            nt["id"] = x["id"]
            choices.append({"note_id": x["id"], "audio_path": nt["audio"], "audio_id": nt["audio_id"], "region_id": nt["region"], "alias": nt["alias"]})
    ok, t = m.call("replace_note_material", roots=[str(LIB)], choices=choices,
                   game_model_dir=str(cfg.MODELS / "game" / "medium"), fcpe_model=str(cfg.MODELS / "fcpe" / "fcpe.onnx"))
    if not ok: raise RuntimeError("replace_note_material: " + t[:400])
    print("bound:", json.loads(t).get("notes"), "notes on", len(lanes), "tracks")
    for nt in notes:
        m.call("edit_notes_pitch", note_id=nt["id"], action="set", midi=nt["midi"])
        m.call("set_note", note_id=nt["id"], drift=nt["drift"], modulation=nt["modulation"], formant_semitones=nt["formant"], gain=round(nt["gain"], 5))
    wav = out / f"teio_rebuilt_{settings.get('pitch_algorithm')}.wav"
    ok, t = m.call("export_wav", path=str(wav), timeout_seconds=600)
    print("export:", ok, t[-120:])
    m.call("project_save", path=str(out / "teio_rebuilt.hjpx"))
    json.dump(notes, open(out / "notes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    m.close()


if __name__ == "__main__":
    main()
