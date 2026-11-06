"""用 DiffSinger 声库的音高模型（只做音高预测，不合成声音）给任务生成音高线。

    python ds_pitch.py <task.json> <voicebank_dir> --speakers standard phantom --lang ja --out <dir>

调用 HachiShifter 自带的 engines/diffsinger/bridge.py（operation=pitch），用完整运行包里的 Python。
输出 <out>/<speaker>.json：{note_index: [[音符内秒数, midi], ...]}
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "annotator"))
import otomad_config as cfg  # noqa: E402
from kana import to_romaji  # noqa: E402

PYTHON = cfg.MODELS.parent / "engines" / "python" / "python.exe"
BRIDGE = cfg.REPO_ROOT / "engines" / "diffsinger" / "bridge.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("voicebank")
    ap.add_argument("--speakers", nargs="+", default=[""])
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--lyric", choices=["kana", "romaji"], default="kana")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocoders", default=str(HERE.parent / "voicebanks" / "vocoders"), help="通用声码器目录（只为通过预检，音高预测不用）")
    a = ap.parse_args()
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    spb = 60.0 / task["bpm"]
    notes = []
    for i, m in enumerate(task["melody"]):
        lyric = m.get("lyric", "a")
        if a.lyric == "romaji" and a.lang == "ja":
            lyric = to_romaji(lyric) or lyric
        notes.append({"id": str(i), "start": m["start_beat"] * spb, "duration": m["beats"] * spb,
                      "lyric": lyric, "midi": m["midi"]})
    for cur, nxt in zip(notes, notes[1:]):  # 单声部：结尾不超过下一个音的开头
        cur["duration"] = min(cur["duration"], nxt["start"] - cur["start"] - 1e-7)
    total = max(n["start"] + n["duration"] for n in notes) + 1.0
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for spk in a.speakers:
        req = {"operation": "pitch", "voicebank": str(Path(a.voicebank).resolve()), "language": a.lang, "speaker": spk,
               "notes": notes, "duration": total, "dictionary": "", "vocoders": str(Path(a.vocoders).resolve())}
        rq, rs = out / f"req_{spk or 'default'}.json", out / f"res_{spk or 'default'}.json"
        rq.write_text(json.dumps(req, ensure_ascii=False), encoding="utf-8")
        t0 = time.time()
        p = subprocess.run([str(PYTHON), "-I", str(BRIDGE), str(rq), str(rs)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(BRIDGE.parent))
        res = json.loads(rs.read_text(encoding="utf-8")) if rs.exists() else {"ok": False, "error": p.stderr[-800:]}
        if not res.get("ok"):
            print(spk, "失败:", res.get("error"), p.stderr[-500:])
            continue
        curves = {c["id"]: c["points"] for c in res["curves"]}
        (out / f"{spk or 'default'}.json").write_text(json.dumps(curves), encoding="utf-8")
        print(f"{spk or 'default'}: {len(curves)}/{len(notes)} 个音符有音高线，{res.get('generated_phrases')} 个乐句，"
              f"{time.time() - t0:.0f}s，{res.get('backend')}")


if __name__ == "__main__":
    main()
