"""从 OpenUtau USTX 截取一条音轨的一段生成调音任务。

    python make_task_from_ustx.py <ustx> --part 0 --seconds 60 --id t10_wonder --roots "日语人力素材/单字" --lang ja

- 时间整体平移到第一个音符所在小节的开头（保留小节内的相对位置和所有休止）
- 每个音符带上 USTX 里的歌词（任务里预填，Agent 不需要再分配）
- USTX 只用于本地测试，任务 json 不入库（歌词有版权）
"""
import argparse
import json
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ustx")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--lang", default="ja")
    a = ap.parse_args()
    d = yaml.safe_load(open(a.ustx, encoding="utf-8-sig"))
    bpm = float(d["bpm"])
    res = int(d.get("resolution", 480))
    bar = res * int(d.get("beat_per_bar", 4))
    part = d["voice_parts"][a.part]
    notes = sorted(part["notes"], key=lambda n: n["position"])
    tick0 = part["position"]
    sec = lambda t: (tick0 + t) / res * 60 / bpm  # noqa: E731
    notes = [n for n in notes if sec(n["position"]) < a.seconds]
    origin = (tick0 + notes[0]["position"]) // bar * bar
    melody = [{"start_beat": round((tick0 + n["position"] - origin) / res, 4), "beats": round(n["duration"] / res, 4),
               "midi": int(n["tone"]), "lyric": str(n["lyric"])} for n in notes]
    task = {"id": a.id, "title": a.title or f"{Path(a.ustx).stem} main 前 {a.seconds:.0f} 秒", "language": a.lang,
            "bpm": bpm, "lyrics": "（每个音符已预填歌词）", "melody": melody, "roots": a.roots,
            "source": {"ustx": str(a.ustx), "part": a.part, "origin_tick": origin, "seconds": a.seconds},
            "brief": "日语单音拍歌声切片人力：歌词已按音符填好；每个发音有多个版本，挑移调最少、辅音不过长的版本。"}
    out = Path(__file__).resolve().parent / "tasks_local" / f"{a.id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(task, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out, len(melody), "notes; 起点小节偏移", origin / bar, "; 时长", round((melody[-1]["start_beat"] + melody[-1]["beats"]) * 60 / bpm, 1), "s")


if __name__ == "__main__":
    main()
