"""用人工标注的 P 点评估各对齐器。

    python evaluate.py out_hfa [out_sofa ...]

P 的取法：phones 层第一个“声母音素”的结束时刻；没有声母（零声母）则取第一个非静音音素的开始。
同时给出规则法 low_rise6（results_v2.json）作为基线。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "pilot"
INITIALS = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r",
            "z", "c", "s", "y", "w"}
try:
    INITIALS |= set(json.loads((Path(__file__).resolve().parent / "mfa_initials.json").read_text(encoding="utf-8")))
except FileNotFoundError:
    pass
SILENCE = {"", "SP", "AP", "sil", "sp", "spn", "<eps>", "EP"}


def phones(tg: Path):
    text = tg.read_text(encoding="utf-8", errors="ignore")
    tier = text.split('name = "phones"', 1)[-1]
    out = []
    for xmin, xmax, lab in re.findall(r'xmin = ([\d.e-]+)\s+xmax = ([\d.e-]+)\s+text = "(.*?)"', tier):
        out.append((float(xmin), float(xmax), lab))
    return out


def p_from(ph):
    seq = [p for p in ph if p[2] not in SILENCE]
    if not seq:
        return None
    first = seq[0]
    if first[2] in INITIALS and len(seq) > 1:
        return first[1]
    return first[0]


def report(name, errs):
    e = np.array(errs)
    if len(e) == 0:
        print(f"{name:12s} 无结果")
        return
    print(f"{name:12s} n={len(e):2d}  中位差 {np.median(e):+6.1f} ms  |差|中位 {np.median(np.abs(e)):5.1f}  "
          f"±15ms {np.mean(np.abs(e) <= 15):4.0%}  ±30ms {np.mean(np.abs(e) <= 30):4.0%}")


def main():
    labels = json.loads((PILOT / "labels_p_single.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in json.loads((PILOT / "results_v2.json").read_text(encoding="utf-8"))}
    good = {k: v["p"] for k, v in labels.items() if not v.get("bad")}
    report("规则 rise6", [(p - rows[k]["cands"]["low_rise6"]) * 1000 for k, p in good.items() if rows[k]["cands"].get("low_rise6") is not None])
    detail = {k: {"file": rows[k]["file"], "你": round(p * 1000)} for k, p in good.items()}
    for out in sys.argv[1:]:
        errs = []
        for k, p in good.items():
            tg = HERE / out / f"{k}.TextGrid"
            if not tg.exists():
                continue
            pp = p_from(phones(tg))
            if pp is None:
                continue
            errs.append((p - pp) * 1000)
            detail[k][out] = round(pp * 1000)
        report(out, errs)
    if "-v" in sys.argv or True:
        print()
        for k, d in sorted(detail.items()):
            print(k, d)


if __name__ == "__main__":
    main()
