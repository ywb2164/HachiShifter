"""把原唱（分离人声）的音高线画到 Agent 做好的人力工程上，并试衔接选项，渲染对比版本。

    python render_with_reference.py <agent run dir> --task ../agent/tasks_local/t10_wonder.json

- 原唱逐帧音高：GAME+FCPE 分析快照（wonder/analysis_snapshot.json），时间偏移见 wonder/alignment.json
- 每个音符：取对应时间段的原唱音高，丢掉偏离乐谱音高 > 2 个半音的点（和声/残留），
  可信点 < 40% 的音保持拉平
- 版本：flat（原样）/ curve（画线）/ curve+smooth（再开 smooth_overlaps）
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import otomad_config as cfg  # noqa: E402
from compare_reference import HOP, f0_track  # noqa: E402
from mcp_client import HachiMcp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=str(HERE / "wonder" / "renders"))
    a = ap.parse_args()
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    bpm = task["bpm"]
    origin = task["source"]["origin_tick"] / 480 * 60 / bpm          # 任务时间 0 = USTX 的这一时刻
    off = json.load(open(HERE / "wonder" / "alignment.json"))["offset_seconds"]  # 音频 = USTX + off
    f0, _ = f0_track(json.load(open(HERE / "wonder" / "analysis_snapshot.json", encoding="utf-8")), 400)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def open_project():
        m = HachiMcp(str(cfg.EXE), [f"--roots={cfg.MIRROR}"])
        ok, t = m.call("project_open", path=str(Path(a.run) / "result.hjpx"))
        assert ok, t
        tr = m.call_json("project_snapshot")["tracks"][-1]
        notes = sorted([n for c in tr["clips"] for n in c["notes"] if n.get("label") != "R"], key=lambda n: n["start_seconds"])
        clip_off = tr["clips"][0].get("start_seconds", 0.0)
        return m, tr, notes, clip_off

    def draw_curves(m, notes, clip_off):
        used = flat = 0
        for n in notes:
            d, base = n["duration_seconds"], n["midi"]
            a0 = clip_off + n["start_seconds"] + origin + off
            pts = []
            for t in np.arange(0, d, HOP):
                k = int(round((a0 + t) / HOP))
                v = f0[k] if 0 <= k < len(f0) else np.nan
                if not np.isnan(v) and abs(v - base) <= 2.0:
                    pts.append({"time_seconds": round(float(t), 3), "midi": round(float(v), 3)})
            if len(pts) >= max(2, 0.4 * d / HOP):
                m.call("set_pitch_curve", note_id=n["id"], points=pts)
                used += 1
            else:
                flat += 1
        return used, flat

    versions = {}
    for name in ("flat", "curve", "curve_smooth"):
        m, tr, notes, clip_off = open_project()
        m.call("set_track", track_id=tr["id"], pitch_algorithm="world", volume=0.8, smooth_overlaps=(name == "curve_smooth"))
        info = {}
        if name != "flat":
            info["curve_notes"], info["flat_notes"] = draw_curves(m, notes, clip_off)
        wav = out / f"{name}.wav"
        ok, t = m.call("export_wav", path=str(wav), timeout_seconds=900)
        m.close()
        y, sr = sf.read(wav)
        info.update(ok=ok, md5=hashlib.md5(wav.read_bytes()).hexdigest(), peak=round(float(np.abs(y).max()), 3))
        versions[name] = info
        print(name, info)
    print("不同的文件数", len({v["md5"] for v in versions.values()}))
    (out / "versions.json").write_text(json.dumps(versions, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
