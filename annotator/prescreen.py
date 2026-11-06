"""素材初审：短辅音版本 + 耐处理测试 + 响度。结果写回 library/<子目录>/annotations.json。

    D:/miniconda/envs/otomad/python.exe prescreen.py "日语人力素材/单字" [--shift 4] [--short-cons-ms 60]

1. 短辅音版本：辅音 > short_cons_ms+10 的素材，另登记一套标注（新 audioId），区域起点挪到 P-short_cons_ms，
   只保留紧挨元音的那段辅音。annotations.json 里 variant="short_consonant"。
2. 耐处理测试：每个素材在测试工程里放两个音（源音高 ±shift 半音，元音拉长约 2 倍），world + 完全拉平渲染一次，
   测每个音的有声概率 / 谐波占比 → robust（0~1，越高越经得起移调拉伸，越不容易电音/痰音）。
3. loudness_db：素材元音段（P 之后）的 RMS 响度，用于音量平衡。
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import otomad_config as cfg  # noqa: E402
from mcp_client import HachiMcp  # noqa: E402


def fl(x):
    return int(max(0.0, x) * 10000) / 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subdir")
    ap.add_argument("--shift", type=float, default=4.0)
    ap.add_argument("--short-cons-ms", type=float, default=60.0)
    a = ap.parse_args()
    root = cfg.MIRROR / a.subdir
    ann_path = root / "annotations.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    rows = [r for r in ann["rows"] if r.get("variant") != "short_consonant"]
    mcp = HachiMcp(str(cfg.EXE), [f"--roots={cfg.MIRROR}"])

    # ---- 1. 短辅音版本 ----
    short_rows = []
    sc = a.short_cons_ms / 1000.0
    for r in rows:
        if r["consonant_ms"] > a.short_cons_ms + 10:
            start = r["p"] - sc
            aid, rid = "a" + uuid.uuid4().hex, "p" + uuid.uuid4().hex
            row = {"name": r["pinyin"], "region_start_seconds": fl(start), "region_end_seconds": fl(r["end"]),
                   "alignment_seconds": fl(r["p"]), "fixed_duration_seconds": fl(r["p"] - start),
                   "material_audio_id": aid, "material_region_id": rid,
                   "native_provenance": "otomad-short-consonant", "native_confidence": r["confidence"]}
            ok, text = mcp.call("sample_settings_save", audio_path=str(root / r["file"]), rows=[row])
            if ok:
                short_rows.append({**r, "region_id": rid, "start": round(start, 4), "consonant_ms": round(a.short_cons_ms, 1),
                                   "variant": "short_consonant", "short_of": r["region_id"]})
    print(f"短辅音版本：{len(short_rows)} 个（原辅音 > {a.short_cons_ms + 10:.0f} ms）")

    # ---- 2. 耐处理测试：一次渲染 ----
    bpm = 120.0
    tick = lambda s: max(1, round(s / (60 / bpm) * 480))  # noqa: E731
    lines = ["[#VERSION]", "UST Version1.2", "[#SETTING]", f"Tempo={bpm:.2f}", "Tracks=1", "ProjectName=prescreen", "VoiceDir=", "Mode2=True"]
    plan, k = [], 0
    for r in rows:
        src = r.get("median_midi") or 67.0
        dur = float(np.clip(2.0 * r["vowel_ms"] / 1000.0, 0.3, 1.0))
        for sgn in (+1, -1):
            midi = int(round(src + sgn * a.shift))
            lines += [f"[#{k:04d}]", f"Length={tick(0.4)}", "Lyric=R", "NoteNum=60"]; k += 1
            lines += [f"[#{k:04d}]", f"Length={tick(dur)}", "Lyric=a", f"NoteNum={midi}"]; k += 1
            plan.append((r, midi, dur))
    lines.append("[#TRACKEND]")
    work = root / "_prescreen"
    work.mkdir(exist_ok=True)
    (work / "prescreen.ust").write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    mcp.call("project_new")
    ok, text = mcp.call("import_ust", path=str(work / "prescreen.ust"))
    assert ok, text
    tr = mcp.call_json("project_snapshot")["tracks"][-1]
    mcp.call("set_track", track_id=tr["id"], pitch_algorithm="world", volume=0.5)
    notes = sorted([n for c in tr["clips"] for n in c["notes"] if n.get("label", "").upper() != "R"], key=lambda n: n["start_seconds"])
    assert len(notes) == len(plan), (len(notes), len(plan))
    q = mcp.call_json("material_query", roots=[str(root)], lyrics=sorted({r["pinyin"] for r in rows}), limit=200, include_sequences=False)
    by_rid = {c["region_id"]: c for item in q["lyrics"] for c in item["candidates"]}
    choices = []
    for n, (r, midi, dur) in zip(notes, plan):
        c = by_rid.get(r["region_id"])
        if c:
            choices.append({"note_id": n["id"], "audio_path": c["audio_path"], "audio_id": c["audio_id"], "region_id": c["region_id"], "alias": c["alias"]})
    print(f"测试音符 {len(notes)} 个，可替换 {len(choices)} 个，替换中（会对每个素材做一次 GAME+FCPE）…")
    rep = mcp.call_json("replace_note_material", roots=[str(root)], choices=choices,
                        game_model_dir=str(cfg.MODELS / "game" / "medium"), fcpe_model=str(cfg.MODELS / "fcpe" / "fcpe.onnx"))
    print("  替换:", {k2: rep[k2] for k2 in ("changed", "notes", "analysed")})
    for n in notes:
        mcp.call("set_note", note_id=n["id"], drift=0, modulation=0)
    ok, text = mcp.call("export_wav", path=str(work / "prescreen.wav"), timeout_seconds=1800)
    assert ok, text
    mcp.close()

    y, sr = librosa.load(work / "prescreen.wav", sr=22050, mono=True)
    _, vf, vp = librosa.pyin(y, fmin=65, fmax=1500, sr=sr, hop_length=220)
    tt = np.arange(len(vp)) * 220 / sr
    pre = max(0.0, len(y) / sr - (notes[-1]["start_seconds"] + notes[-1]["duration_seconds"]))
    per = {}
    for n, (r, midi, dur) in zip(notes, plan):
        s0, s1 = pre + n["start_seconds"] + 0.03, pre + n["start_seconds"] + n["duration_seconds"] - 0.03
        seg = y[int(s0 * sr):int(s1 * sr)]
        mk = (tt >= s0) & (tt < s1)
        if len(seg) < 2048 or not mk.any():
            continue
        H, _ = librosa.effects.hpss(seg)
        harm = float(np.sum(H ** 2) / (np.sum(seg ** 2) + 1e-12))
        voi = float(np.nanmean(vp[mk]))
        per.setdefault(r["region_id"], []).append((harm, voi))

    # ---- 3. 响度（源素材元音段）+ 汇总 ----
    for r in rows:
        yy, sr2 = librosa.load(root / r["file"], sr=22050, mono=True)
        v = yy[int(r["p"] * sr2):int(r["end"] * sr2)]
        r["loudness_db"] = round(float(20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-9)), 1) if len(v) else None
        m = per.get(r["region_id"])
        if m:
            harm, voi = np.mean([x[0] for x in m]), np.mean([x[1] for x in m])
            r["robust_harmonic"], r["robust_voicing"] = round(float(harm), 3), round(float(voi), 3)
    hs = np.array([r["robust_harmonic"] for r in rows if "robust_harmonic" in r])
    vs = np.array([r["robust_voicing"] for r in rows if "robust_voicing" in r])
    for r in rows:  # 在本库内排名（0~1），避免不同素材库绝对值不可比
        if "robust_harmonic" in r:
            r["robust"] = round(float(0.5 * np.mean(hs <= r["robust_harmonic"]) + 0.5 * np.mean(vs <= r["robust_voicing"])), 3)
    for s in short_rows:  # 短辅音版本继承原素材的耐处理分与响度
        src = next(r for r in rows if r["region_id"] == s["short_of"])
        for key in ("robust", "robust_harmonic", "robust_voicing", "loudness_db"):
            if key in src:
                s[key] = src[key]
    ann["rows"] = rows + short_rows
    ann_path.write_text(json.dumps(ann, ensure_ascii=False, indent=1), encoding="utf-8")
    rb = [r["robust"] for r in rows if "robust" in r]
    print(f"耐处理分：{len(rb)} 个素材；处理后谐波占比 中位 {np.median(hs):.2f}（范围 {hs.min():.2f}~{hs.max():.2f}），有声概率 中位 {np.median(vs):.2f}")
    worst = sorted((r for r in rows if "robust" in r), key=lambda r: r["robust"])[:8]
    print("最不耐处理的 8 个：", [(r["file"], r["robust"]) for r in worst])


if __name__ == "__main__":
    main()
