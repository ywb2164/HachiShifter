"""端到端验证新 MCP 工具：登记素材 → 导入 UST → 查素材 → 替换 → 切分/合并 → 导出 WAV → 检查音高。

用法：python test_material_flow.py "<HachiShifter Next.exe>"
素材库只读：测试用的单字会复制到 otomad-agent/materials_test/gun/，otto.ini 也只写在那里。
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import otomad_config as cfg  # noqa: E402

import json
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pilot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_syllables import analyse, place_p, split_pinyin  # noqa: E402
from mcp_client import HachiMcp  # noqa: E402

LIB = cfg.LIBRARY / "棍の音" / "棍音源"
MODELS = cfg.MODELS
WORK = ROOT / "materials_test"
LYRICS = ["da", "jia", "hao", "wo", "shi", "dian", "gun"]
MIDIS = [60, 62, 64, 65, 67, 69, 71]
NOTE_SEC = 0.5  # 120 BPM 八分音符 = 240 tick；这里用四分音符 480 tick = 0.5 s


def check(ok: bool, what: str) -> None:
    print(("  ✓ " if ok else "  ✗ ") + what)
    if not ok:
        raise SystemExit(1)


def main() -> None:
    exe = sys.argv[1]
    gun = WORK / "gun"
    if WORK.exists():
        shutil.rmtree(WORK)
    gun.mkdir(parents=True)
    mcp = HachiMcp(exe, [f"--roots={WORK}"])
    tools = mcp.tools()
    for t in ("material_query", "replace_note_material", "split_note", "merge_notes"):
        check(t in tools, f"tools/list 包含 {t}")

    print("1. 登记素材（自动 P 点 → otto.ini）")
    for syl in LYRICS:
        src = LIB / f"{syl}.wav"
        dst = gun / src.name
        shutil.copy2(src, dst)
        r = analyse(dst)
        _, ini, kind = split_pinyin(syl)
        place_p(r, kind, ini)
        p = r["p"] if r.get("p") is not None else r["start"]
        row = {"name": syl, "region_start_seconds": r["start"], "region_end_seconds": r["end"],
               "alignment_seconds": p, "fixed_duration_seconds": max(0.0, p - r["start"]),
               "native_provenance": "otomad-auto", "native_confidence": 0.9 if r["verdict"] == "自信" else 0.4}
        ok, text = mcp.call("sample_settings_save", audio_path=str(dst), rows=[row])
        check(ok, f"{syl}: {text[:80]}  P={p:.3f}s 辅音 {r.get('consonant_ms')} ms")
    print("   otto.ini:", (gun / "otto.ini").exists())

    print("2. 导入 UST 乐谱")
    ust = WORK / "test.ust"
    lines = ["[#VERSION]", "UST Version1.2", "[#SETTING]", "Tempo=120.00", "Tracks=1", "ProjectName=otomad-test", "VoiceDir=", "Mode2=True"]
    for i, (lyr, m) in enumerate(zip(LYRICS, MIDIS)):
        lines += [f"[#{i:04d}]", "Length=480", f"Lyric={lyr}", f"NoteNum={m}", "PreUtterance=", "Intensity=100", "Modulation=0"]
    lines.append("[#TRACKEND]")
    ust.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    ok, text = mcp.call("project_new")
    ok, text = mcp.call("import_ust", path=str(ust))
    check(ok, f"import_ust: {text[:120]}")
    snap = mcp.call_json("project_snapshot")
    track = snap["tracks"][-1]
    notes = [n for c in track["clips"] for n in c["notes"]]
    print(f"   轨道 {track.get('name')} type={track.get('track_type')} pitch={track.get('pitch_algorithm')} 音符 {len(notes)}")
    check(len(notes) == len(LYRICS), "音符数与歌词一致")
    if track.get("pitch_algorithm") != "mld5":
        ok, text = mcp.call("set_track", track_id=track["id"], pitch_algorithm="mld5")
        check(ok, f"音高算法切到 mld5: {text[:80]}")

    print("3. material_query")
    q = mcp.call_json("material_query", roots=[str(gun)], lyrics=LYRICS)
    print(f"   索引 {q['entry_count']} 条，warnings={q['warnings'][:2]}")
    for item in q["lyrics"]:
        check(item["total"] >= 1, f"{item['lyric']}: {item['total']} 个候选，P={item['candidates'][0]['preutterance_seconds']:.3f}s")

    print("4. replace_note_material")
    notes.sort(key=lambda n: n["start_seconds"])
    choices = []
    for n, item in zip(notes, q["lyrics"]):
        c = item["candidates"][0]
        choices.append({"note_id": n["id"], "audio_path": c["audio_path"], "audio_id": c["audio_id"],
                        "region_id": c["region_id"], "alias": c["alias"]})
    rep = mcp.call_json("replace_note_material", roots=[str(gun)], choices=choices,
                        game_model_dir=str(MODELS / "game" / "medium"), fcpe_model=str(MODELS / "fcpe" / "fcpe.onnx"))
    print("   ", json.dumps(rep, ensure_ascii=False)[:300])
    check(rep["changed"], "替换已写入工程")

    snap = mcp.call_json("project_snapshot")
    track = next(t for t in snap["tracks"] if t["id"] == track["id"])
    notes = sorted((n for c in track["clips"] for n in c["notes"]), key=lambda n: n.get("start_seconds", 0))
    print(f"   替换后片段 {len(track['clips'])} 个，音符 {len(notes)} 个")

    print("5. split_note / merge_notes / undo")
    target = notes[2]
    sp = mcp.call_json("split_note", note_id=target["id"], local_seconds=target["duration_seconds"] / 2)
    check(isinstance(sp, dict) and sp.get("new_note_id"), f"split_note → {sp}")
    ok, text = mcp.call("merge_notes", note_ids=[sp["note_id"], sp["new_note_id"]])
    print(f"   merge_notes: {ok} {text[:120]}")
    ok, _ = mcp.call("undo")
    ok2, _ = mcp.call("undo") if ok else (False, "")
    print(f"   undo×2: {ok} {ok2}")

    print("6. 导出 WAV 并检查音高")
    out_wav = WORK / "render.wav"
    ok, text = mcp.call("export_wav", path=str(out_wav), timeout_seconds=300)
    check(ok and out_wav.exists(), f"export_wav: {text[:160]}")
    y, sr = librosa.load(out_wav, sr=22050, mono=True)
    f0, v, _ = librosa.pyin(y, fmin=65, fmax=1500, sr=sr, hop_length=220)
    t = np.arange(len(f0)) * 220 / sr
    offset = min(n.get("project_start_seconds", n["start_seconds"]) for n in notes) if notes else 0
    for i, (lyr, m) in enumerate(zip(LYRICS, MIDIS)):
        a, b = i * NOTE_SEC + 0.1, (i + 1) * NOTE_SEC - 0.05
        seg = f0[(t >= a) & (t < b) & v]
        med = float(np.nanmedian(librosa.hz_to_midi(seg))) if len(seg) else float("nan")
        print(f"   {lyr}: 目标 MIDI {m}，渲染中位 {med:.2f}，偏差 {100 * (med - m):+.0f} cents")
    print(f"   (时间偏移 {offset})")
    ok, text = mcp.call("project_save", path=str(WORK / "otomad-test.hjpx"))
    check(ok, f"保存工程 {text}")
    mcp.close()


if __name__ == "__main__":
    main()
