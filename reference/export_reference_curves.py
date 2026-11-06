"""把原唱（分离人声）音高线按任务音符切好，导出成与 ds_pitch.py 相同的格式：{note_index: [[音符内秒数, midi], ...]}。

    python export_reference_curves.py ../agent/tasks_local/t10_wonder.json wonder --out wonder/ref_curves.json
偏离乐谱音高 > 2 半音的点丢弃（和声/残留），可信点不足 40% 的音不给线（保持拉平）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_reference import HOP, f0_track  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("refdir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    spb = 60 / task["bpm"]
    origin = task["source"]["origin_tick"] / 480 * spb
    ref = Path(a.refdir)
    off = json.load(open(ref / "alignment.json"))["offset_seconds"]
    f0, _ = f0_track(json.load(open(ref / "analysis_snapshot.json", encoding="utf-8")), 400)
    out = {}
    for i, m in enumerate(task["melody"]):
        d, base = m["beats"] * spb, m["midi"]
        a0 = m["start_beat"] * spb + origin + off
        pts = []
        for t in np.arange(0, d, HOP):
            k = int(round((a0 + t) / HOP))
            v = f0[k] if 0 <= k < len(f0) else np.nan
            if not np.isnan(v) and abs(v - base) <= 2.0:
                pts.append([round(float(t), 3), round(float(v), 3)])
        if len(pts) >= max(2, 0.4 * d / HOP):
            out[str(i)] = pts
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"{len(out)}/{len(task['melody'])} 个音符有原唱音高线 → {a.out}")


if __name__ == "__main__":
    main()
