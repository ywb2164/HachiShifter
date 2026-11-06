"""衔接对比：原唱音高线 + {不处理 / 两轨交替音尾延长淡出 / +压音头 / 延长更长}。

    python render_legato.py <agent run dir> --task ../agent/tasks_local/t10_wonder.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import otomad_config as cfg  # noqa: E402
from compare_reference import HOP, f0_track  # noqa: E402
from legato import apply_legato, notes_of, split_alternate  # noqa: E402
from mcp_client import HachiMcp  # noqa: E402

SETTINGS = {"pitch_algorithm": "world", "volume": 0.8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=str(HERE / "wonder" / "renders_legato"))
    a = ap.parse_args()
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    origin = task["source"]["origin_tick"] / 480 * 60 / task["bpm"]
    off = json.load(open(HERE / "wonder" / "alignment.json"))["offset_seconds"]
    f0, _ = f0_track(json.load(open(HERE / "wonder" / "analysis_snapshot.json", encoding="utf-8")), 400)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def base_project():
        m = HachiMcp(str(cfg.EXE), [f"--roots={cfg.MIRROR}"])
        ok, t = m.call("project_open", path=str(Path(a.run) / "result.hjpx"))
        assert ok, t
        tr, clip, notes = notes_of(m, m.call_json("project_snapshot")["tracks"][-1]["id"])
        m.call("set_track", track_id=tr["id"], smooth_overlaps=True, **SETTINGS)
        coff = clip.get("start_seconds", 0.0)
        for n in notes:
            if n.get("label", "").upper() == "R":
                continue
            d, base = n["duration_seconds"], n["midi"]
            a0 = coff + n["start_seconds"] + origin + off
            pts = []
            for t in np.arange(0, d, HOP):
                k = int(round((a0 + t) / HOP))
                v = f0[k] if 0 <= k < len(f0) else np.nan
                if not np.isnan(v) and abs(v - base) <= 2.0:
                    pts.append({"time_seconds": round(float(t), 3), "midi": round(float(v), 3)})
            if len(pts) >= max(2, 0.4 * d / HOP):
                m.call("set_pitch_curve", note_id=n["id"], points=pts)
        return m, tr["id"], coff, [n for n in notes if n.get("label", "").upper() != "R"]

    def export(m, name):
        wav = out / f"{name}.wav"
        ok, t = m.call("export_wav", path=str(wav), timeout_seconds=900)
        m.close()
        y, _ = sf.read(wav)
        info = {"ok": ok, "md5": hashlib.md5(wav.read_bytes()).hexdigest(), "peak": round(float(np.abs(y).max()), 3)}
        print(name, info)
        return info

    versions = {}
    m, tid, coff, notes = base_project()
    versions["base"] = export(m, "base")

    # 测音头：音头（辅音起点到音符起点后 40 ms）响度 vs 元音主体
    y, sr = librosa.load(out / "base.wav", sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=441, hop_length=110)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    fr = lambda t: int(t * sr / 110)  # noqa: E731
    head_cut = {}
    for k, n in enumerate(notes):
        s = coff + n["start_seconds"]
        cons = float(n.get("consonant_seconds") or 0.0)
        h0, h1 = fr(s - cons), fr(s + 0.04)
        b0, b1 = fr(s + 0.06), fr(s + n["duration_seconds"] - 0.02)
        if b1 - b0 < 3 or h1 <= h0:
            continue
        excess = float(db[h0:h1].max() - np.median(db[b0:b1]))
        if excess > 4.0:
            head_cut[k] = round(min(8.0, excess - 2.0), 1)
    print(f"音头偏响（>4 dB）的音符 {len(head_cut)} 个，平均压 {np.mean(list(head_cut.values())) if head_cut else 0:.1f} dB")

    for name, tail, heads in (("legato60", 60, None), ("legato60_head", 60, head_cut), ("legato120", 120, None)):
        m, tid, coff, _ = base_project()
        m.call("set_track", track_id=tid, smooth_overlaps=False)
        a_tr, b_tr = split_alternate(m, tid, dict(SETTINGS, smooth_overlaps=False))
        info = apply_legato(m, a_tr, b_tr, tail_ms=tail, head_cut=heads)
        print(" ", name, info)
        versions[name] = {**export(m, name), **info}
    print("不同的文件数", len({v["md5"] for v in versions.values()}))
    (out / "versions.json").write_text(json.dumps({**versions, "head_cut": head_cut}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
