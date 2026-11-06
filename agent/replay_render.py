"""同一套素材（某次运行记录的 picks）+ 同样的音高线/音量/衔接，只换渲染算法设置。

    python replay_render.py runs/<运行目录> --task tasks_local/t10_wonder.json --name mld5_user \
        --render '{"pitch_algorithm":"mld5","stretch_algorithm":"melodyne-hybrid","render_order":"process-then-splice","merge_contiguous_source":true}'
输出 runs/<运行目录>/replay/<name>/，结果追加到 replay/variants.json。
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import tools
from tools import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--task", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--render", required=True, help="set_track 参数 JSON")
    ap.add_argument("--tail-ratio", type=float, default=0.35)
    ap.add_argument("--pitch", default="run", help="run=沿用运行时的音高线来源；none=不画线，只把音符设到 乐谱+定音偏移")
    ap.add_argument("--drift", type=float, default=None, help="给定则覆盖拉平（默认流程是 0）")
    ap.add_argument("--modulation", type=float, default=None)
    ap.add_argument("--no-balance", action="store_true", help="不做逐音音量平衡")
    ap.add_argument("--no-legato", action="store_true", help="不做两轨交替衔接")
    ap.add_argument("--save", default=None, help="渲染后另存工程 .hjpx")
    ap.add_argument("--ref-audio", default=None, help="另存前加一条静音参考轨（从 0 秒开始的原唱片段）")
    a = ap.parse_args()
    run = Path(a.run)
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    tools.RENDER.clear()
    tools.RENDER.update(json.loads(a.render))  # Session 建轨和 legato 复制轨都读这个全局
    out = run / "replay"
    out.mkdir(exist_ok=True)
    d = out / a.name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    s = Session(task, d)
    s.apply_picks(summ["picks"])
    if a.pitch == "none":
        tune = s._tune()
        for n in s.notes:
            if n.get("material"):
                s.mcp.call("edit_notes_pitch", note_id=n["id"], action="set", midi=n["midi"] + tune)
                n["target"] = n["midi"] + tune
        s.pitch_source = "none"
    else:
        s.apply_pitch_curves(summ.get("pitch_source") or "reference")
    if a.drift is not None or a.modulation is not None:
        kw = {k: v for k, v in (("drift", a.drift), ("modulation", a.modulation)) if v is not None}
        for n in s.notes:
            if n.get("material"):
                s.mcp.call("set_note", note_id=n["id"], **kw)
    s.render_and_check()
    if not a.no_balance:
        s.balance_loudness()
    if not a.no_legato:
        s.finalize_legato(tail_ratio=a.tail_ratio, tail_min_ms=20.0)
    chk = s.render_and_check()
    snap = s.mcp.call_json("project_snapshot")
    notes_sample = [{k: n.get(k) for k in ("midi", "drift", "modulation", "flattened")} | {"curve_points": len(n.get("pitch_control_points") or [])}
                    for n in snap["tracks"][0]["clips"][0]["notes"][:4]]
    if a.save:
        if a.ref_audio:
            before = {t["id"] for t in snap["tracks"]}
            ok, text = s.mcp.call("import_audio", path=a.ref_audio, start_seconds=0.0)
            ref = next(t for t in s.mcp.call_json("project_snapshot")["tracks"] if t["id"] not in before)
            s.mcp.call("set_track", track_id=ref["id"], name="原唱参考(静音)", compose=False, muted=True)
        ok, text = s.mcp.call("project_save", path=str(Path(a.save).resolve()))
        print("保存:", text)
    tracks = [{k: t.get(k) for k in ("name", "pitch_algorithm", "stretch_algorithm", "render_order", "merge_contiguous_source")}
              for t in s.mcp.call_json("project_snapshot")["tracks"]]
    s.close()
    wav = Path(chk["render"])
    vj = out / "variants.json"
    result = json.loads(vj.read_text(encoding="utf-8")) if vj.exists() else {}
    result[a.name] = {"wav": str(wav), "score": chk["score"], "md5": hashlib.md5(wav.read_bytes()).hexdigest(),
                      "render": tools.RENDER, "tracks": tracks, "pitch": a.pitch, "drift": a.drift, "modulation": a.modulation,
                      "balance": not a.no_balance, "legato": not a.no_legato, "project": a.save,
                      "check": {k: chk.get(k) for k in ("in_tune_ratio", "harmonic_ratio", "voicing_mean")},
                      "notes_sample": notes_sample}
    vj.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result[a.name], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
