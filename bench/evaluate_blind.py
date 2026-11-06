"""盲标评测：40 个新样本比各方法；10 个重复样本测你自己两次标注的一致性。

    python evaluate_blind.py
HubertFA 偏移量取第一轮（非盲标）估计值，与盲标数据独立。
"""
import json
from pathlib import Path

import numpy as np

from evaluate import phones, p_from

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "pilot"
HFA_OFFSET_MS = 14.0  # 第一轮 28 个标注的中位差


def stat(name, e):
    e = np.asarray([x for x in e if x is not None and np.isfinite(x)])
    if len(e) == 0:
        print(f"  {name:26s} 无数据")
        return
    print(f"  {name:26s} n={len(e):2d}  中位差 {np.median(e):+6.1f} ms  |差|中位 {np.median(np.abs(e)):5.1f}  "
          f"±15ms {np.mean(np.abs(e) <= 15):4.0%}  ±30ms {np.mean(np.abs(e) <= 30):4.0%}")


def tg_p(folder, bid):
    f = HERE / folder / f"{bid}.TextGrid"
    if not f.exists():
        return None
    p = p_from(phones(f))
    return None if p is None else p * 1000


def main():
    items = {it["id"]: it for it in json.loads((PILOT / "blind.json").read_text(encoding="utf-8"))}
    blind = json.loads((PILOT / "labels_p_blind.json").read_text(encoding="utf-8"))
    first = json.loads((PILOT / "labels_p_single.json").read_text(encoding="utf-8"))
    rule = json.loads((HERE / "blind_rule.json").read_text(encoding="utf-8"))
    done = {k: v for k, v in blind.items() if not v.get("bad")}
    print(f"盲标 {len(blind)} 个，其中素材问题 {sum(1 for v in blind.values() if v.get('bad'))}")

    new = [k for k in done if items[k]["repeat_of"] is None]
    print(f"\n【新样本 {len(new)} 个】各方法与你的差（你 − 方法）")
    you = {k: done[k]["p"] * 1000 for k in new}
    stat("规则 low_rise6", [you[k] - rule[k]["low_rise6"] * 1000 if rule[k].get("low_rise6") is not None else None for k in new])
    stat("旧规则 pyin", [you[k] - rule[k]["pyin_onset"] * 1000 if rule[k].get("pyin_onset") is not None else None for k in new])
    h = {k: tg_p("blind_hfa", k) for k in new}
    stat("HubertFA 原始", [you[k] - h[k] if h[k] is not None else None for k in new])
    stat(f"HubertFA +{HFA_OFFSET_MS:.0f}ms", [you[k] - h[k] - HFA_OFFSET_MS if h[k] is not None else None for k in new])
    s = {k: tg_p("blind_sofa", k) for k in new}
    stat("SOFA", [you[k] - s[k] if s[k] is not None else None for k in new])
    if (HERE / "blind_mfa").exists():
        m = {k: tg_p("blind_mfa", k) for k in new}
        stat("MFA", [you[k] - m[k] if m[k] is not None else None for k in new])
    comb = []
    for k in new:
        r, hh = rule[k].get("low_rise6"), h[k]
        if r is not None and hh is not None:
            comb.append(you[k] - (r * 1000 + hh + HFA_OFFSET_MS) / 2)
    stat("规则与HubertFA平均", comb)

    rep = [k for k in done if items[k]["repeat_of"] is not None]
    print(f"\n【重复样本 {len(rep)} 个】你这次 − 你上次（人耳自身一致性）")
    diffs = []
    for k in rep:
        old = first.get(items[k]["repeat_of"], {})
        if old.get("bad") or "p" not in old:
            continue
        d = (done[k]["p"] - old["p"]) * 1000
        diffs.append(d)
        print(f"    {items[k]['file']:12s} 上次 {old['p'] * 1000:6.0f}  这次 {done[k]['p'] * 1000:6.0f}  差 {d:+5.0f} ms"
              + ("（上次是直接接受黄线）" if old.get("same") else ""))
    stat("你两次之间", diffs)


if __name__ == "__main__":
    main()
