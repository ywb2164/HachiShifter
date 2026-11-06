"""把 DiffSinger 生成的音高线画到 Agent 做好的人力工程上，渲染对比版本（全部开 smooth_overlaps）。

    python render_ds_curves.py <agent run dir> --curves <ds_pitch 输出目录> --speakers standard phantom lilith hunter
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
import otomad_config as cfg  # noqa: E402
from mcp_client import HachiMcp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--curves", required=True)
    ap.add_argument("--speakers", nargs="+", required=True)
    ap.add_argument("--out", default=str(HERE / "wonder" / "renders_ds"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ["flat"] + [f"ds_{s}" for s in a.speakers]:
        m = HachiMcp(str(cfg.EXE), [f"--roots={cfg.MIRROR}"])
        ok, t = m.call("project_open", path=str(Path(a.run) / "result.hjpx"))
        assert ok, t
        tr = m.call_json("project_snapshot")["tracks"][-1]
        notes = sorted([n for c in tr["clips"] for n in c["notes"] if n.get("label") != "R"], key=lambda n: n["start_seconds"])
        m.call("set_track", track_id=tr["id"], pitch_algorithm="world", volume=0.8, smooth_overlaps=True)
        info = {}
        if name.startswith("ds_"):
            curves = json.loads((Path(a.curves) / f"{name[3:]}.json").read_text(encoding="utf-8"))
            drawn = 0
            for i, n in enumerate(notes):
                pts = curves.get(str(i))
                if not pts:
                    continue
                d = n["duration_seconds"]
                pts = [{"time_seconds": round(float(t), 3), "midi": round(float(v), 3)} for t, v in pts if 0 <= t <= d]
                if len(pts) >= 2:
                    m.call("set_pitch_curve", note_id=n["id"], points=pts)
                    drawn += 1
            info["drawn"] = drawn
        wav = out / f"{name}.wav"
        ok, t = m.call("export_wav", path=str(wav), timeout_seconds=900)
        m.close()
        y, _ = sf.read(wav)
        info.update(ok=ok, md5=hashlib.md5(wav.read_bytes()).hexdigest(), peak=round(float(np.abs(y).max()), 3))
        versions[name] = info
        print(name, info)
    print("不同的文件数", len({v["md5"] for v in versions.values()}))
    (out / "versions.json").write_text(json.dumps(versions, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
