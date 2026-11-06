"""同一套素材（来自某次运行记录的 picks），只改收尾衔接方式，渲染对照版本。

    python replay_variants.py runs/<运行目录> --task tasks_local/t10_wonder.json
流程与 Agent 一致：放素材 → 画音高线（运行时用的来源）→ 渲染 → 平衡音量 → 衔接（各版本不同）→ 渲染。
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tools import Session

VARIANTS = {"ratio35": dict(tail_ratio=0.35, tail_min_ms=20.0), "fixed60": dict(tail_ratio=None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    out = run / "replay"
    out.mkdir(exist_ok=True)
    result = {}
    for name, kw in VARIANTS.items():
        d = out / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        s = Session(task, d)
        s.apply_picks(summ["picks"])
        s.apply_pitch_curves(summ.get("pitch_source") or "reference")
        s.render_and_check()
        s.balance_loudness()
        if kw.get("tail_ratio") is None:  # v4 做法：固定 60 ms
            from legato import apply_legato, split_alternate
            from tools import RENDER
            a_tr, b_tr = split_alternate(s.mcp, s.track_id, dict(RENDER, volume=s.volume))
            apply_legato(s.mcp, a_tr, b_tr, tail_ms=60.0)
            s.legato_done = True
        else:
            s.finalize_legato(**kw)
        chk = s.render_and_check()
        s.close()
        wav = Path(chk["render"])
        result[name] = {"wav": str(wav), "score": chk["score"], "md5": hashlib.md5(wav.read_bytes()).hexdigest()}
        print(name, result[name])
    (out / "variants.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
