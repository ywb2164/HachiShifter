"""用标注好的真实素材做一段 demo：UST 乐谱 → material_query → 按置信度选素材 → 替换 → 压平音高 → 导出 WAV。

    python render_demo.py
输出 otomad-agent/demo/demo.wav 和 demo.hjpx（可用开发版 HachiShifter 打开）。
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import otomad_config as cfg  # noqa: E402

import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcp_client import HachiMcp  # noqa: E402

AGENT = Path(__file__).resolve().parents[1]
EXE = str(cfg.EXE)
ROOTS = [AGENT / "library" / "棍の音" / "棍音源"]
OUT = AGENT / "demo"
# 简单原创旋律（C 大调），每拍一个字，120 BPM
NOTES = [("da", 60, 480), ("jia", 62, 480), ("hao", 64, 960), ("wo", 64, 480), ("shi", 62, 480),
         ("dian", 60, 480), ("gun", 62, 960), ("ni", 64, 480), ("men", 65, 480), ("hao", 67, 960)]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ust = OUT / "demo.ust"
    lines = ["[#VERSION]", "UST Version1.2", "[#SETTING]", "Tempo=120.00", "Tracks=1", "ProjectName=otomad-demo", "VoiceDir=", "Mode2=True"]
    for i, (lyr, m, ln) in enumerate(NOTES):
        lines += [f"[#{i:04d}]", f"Length={ln}", f"Lyric={lyr}", f"NoteNum={m}", "Intensity=100", "Modulation=0"]
    lines.append("[#TRACKEND]")
    ust.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")

    mcp = HachiMcp(EXE, [f"--roots={AGENT / 'library'}"])
    mcp.call("project_new")
    ok, text = mcp.call("import_ust", path=str(ust))
    assert ok, text
    snap = mcp.call_json("project_snapshot")
    track = snap["tracks"][-1]
    mcp.call("set_track", track_id=track["id"], pitch_algorithm="mld5")
    notes = sorted((n for c in track["clips"] for n in c["notes"]), key=lambda n: n["start_seconds"])

    q = mcp.call_json("material_query", roots=[str(r) for r in ROOTS], lyrics=[n[0] for n in NOTES], limit=10)
    choices = []
    for n, item in zip(notes, q["lyrics"]):
        if not item["candidates"]:
            print("  没有素材:", item["lyric"])
            continue
        c = max(item["candidates"], key=lambda c: c.get("native_confidence", 0))
        choices.append({"note_id": n["id"], "audio_path": c["audio_path"], "audio_id": c["audio_id"],
                        "region_id": c["region_id"], "alias": c["alias"]})
        print(f"  {item['lyric']:5s} ← {Path(c['audio_path']).name}  P={c['preutterance_seconds'] * 1000:.0f} ms  置信 {c.get('native_confidence', 0):.2f}")
    models = cfg.MODELS
    rep = mcp.call_json("replace_note_material", roots=[str(r) for r in ROOTS], choices=choices,
                        game_model_dir=str(models / "game" / "medium"), fcpe_model=str(models / "fcpe" / "fcpe.onnx"))
    print("  替换:", json.dumps(rep, ensure_ascii=False))
    for n in notes:
        mcp.call("set_note", note_id=n["id"], drift=0, modulation=0)
    ok, text = mcp.call("export_wav", path=str(OUT / "demo.wav"), timeout_seconds=300)
    assert ok, text
    mcp.call("project_save", path=str(OUT / "demo.hjpx"))
    mcp.close()

    y, sr = librosa.load(OUT / "demo.wav", sr=22050, mono=True)
    f0, v, _ = librosa.pyin(y, fmin=65, fmax=1500, sr=sr, hop_length=220)
    t = np.arange(len(f0)) * 220 / sr
    pos = 0.0
    for lyr, m, ln in NOTES:
        dur = ln / 480 * 0.5
        seg = f0[(t >= pos + 0.08) & (t < pos + dur - 0.05) & v]
        med = float(np.nanmedian(librosa.hz_to_midi(seg))) if len(seg) else float("nan")
        print(f"  {lyr:5s} 目标 {m}  渲染 {med:6.2f}  偏差 {100 * (med - m):+4.0f} cents")
        pos += dur
    print("输出:", OUT / "demo.wav")


if __name__ == "__main__":
    main()
