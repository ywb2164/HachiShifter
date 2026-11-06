"""第三轮：drift / modulation 分开调，MLD5 与 world 对比。统一音量 0.7 避免削波干扰。

    python render_variants3.py
输出 demo/round3/<编号>.wav、variants.json；渲染后先查重、测峰值。
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import otomad_config as cfg  # noqa: E402
from mcp_client import HachiMcp  # noqa: E402

AGENT = cfg.AGENT_ROOT
DEMO = AGENT / "demo" / "demo.hjpx"
OUT = AGENT / "demo" / "round3"
COMBOS = [("mld5", 0, 0), ("mld5", 0, 0.5), ("mld5", 0, 1.0), ("mld5", 0.2, 1.0), ("world", 0, 0), ("world", 0, 1.0)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    combos = COMBOS[:]
    random.Random(11).shuffle(combos)
    mapping = {}
    for i, (algo, drift, mod) in enumerate(combos, 1):
        mcp = HachiMcp(str(cfg.EXE), [f"--roots={AGENT / 'library'}"])
        ok, text = mcp.call("project_open", path=str(DEMO))
        assert ok, text
        track = mcp.call_json("project_snapshot")["tracks"][-1]
        ok, text = mcp.call("set_track", track_id=track["id"], pitch_algorithm=algo, volume=0.7)
        assert ok, text
        for c in track["clips"]:
            for n in c["notes"]:
                mcp.call("set_note", note_id=n["id"], drift=drift, modulation=mod)
        wav = OUT / f"{i}.wav"
        ok, text = mcp.call("export_wav", path=str(wav), timeout_seconds=600)
        mcp.close()
        y, _ = sf.read(wav)
        md5 = hashlib.md5(wav.read_bytes()).hexdigest()
        mapping[str(i)] = {"algorithm": algo, "drift": drift, "modulation": mod, "md5": md5,
                           "peak": round(float(np.abs(y).max()), 3)}
        print(i, algo, drift, mod, "OK" if ok else text[:120], "峰值", mapping[str(i)]["peak"], md5[:8])
    distinct = len({m["md5"] for m in mapping.values()})
    print(f"{len(mapping)} 个渲染，互不相同的 {distinct} 个")
    (OUT / "variants.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
