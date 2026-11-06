"""同一段 demo，用不同渲染算法 × 音高保留程度各渲染一版，供盲听比较。

    python render_variants.py
输出 otomad-agent/demo/variants/<编号>.wav 和 variants.json（编号 → 设置，打乱顺序，盲听时不显示）。
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import otomad_config as cfg  # noqa: E402

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcp_client import HachiMcp  # noqa: E402

AGENT = Path(__file__).resolve().parents[1]
EXE = str(cfg.EXE)
DEMO = AGENT / "demo" / "demo.hjpx"
OUT = AGENT / "demo" / (sys.argv[1] if len(sys.argv) > 1 else "variants")
ROUND2 = len(sys.argv) > 1 and sys.argv[1] == "round2"
ALGOS = ["world"] if ROUND2 else ["mld5", "nsf-hifigan", "world", "llsm2"]
PITCH = {"拉平": {"drift": 0, "modulation": 0}} if ROUND2 else {"拉平": {"drift": 0, "modulation": 0}, "保留一半原音起伏": {"drift": 0.5, "modulation": 0.5}}
STRETCH = (["melodyne-hybrid", "variable-mel-hop", "loop", "soundtouch", "nsf-shift-then-splice", "hifishifter-mel", "wcsndm-nonloop"]
           if ROUND2 else [None])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    combos = [(a, p, st) for a in ALGOS for p in PITCH for st in STRETCH]
    random.Random(7).shuffle(combos)
    mapping = {}
    for i, (algo, pname, stretch) in enumerate(combos):
        vid = chr(ord("A") + i)
        mcp = HachiMcp(EXE, [f"--roots={AGENT / 'library'}"])
        ok, text = mcp.call("project_open", path=str(DEMO))
        assert ok, text
        snap = mcp.call_json("project_snapshot")
        track = snap["tracks"][-1]
        extra = {"stretch_algorithm": stretch} if stretch else {}
        ok, text = mcp.call("set_track", track_id=track["id"], pitch_algorithm=algo, **extra)
        if not ok:
            print(vid, algo, pname, "设置失败:", text[:120])
            mcp.close()
            continue
        for c in track["clips"]:
            for n in c["notes"]:
                mcp.call("set_note", note_id=n["id"], **PITCH[pname])
        ok, text = mcp.call("export_wav", path=str(OUT / f"{vid}.wav"), timeout_seconds=600)
        backend = ""
        try:
            backend = json.loads(text).get("backend", "")
        except Exception:
            pass
        print(vid, algo, pname, stretch, "OK" if ok else "失败: " + text[:200], backend[:60])
        if ok:
            mapping[vid] = {"algorithm": algo, "pitch": pname, "stretch": stretch, "backend": backend}
        mcp.close()
    (OUT / "variants.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
