"""用 USTX 当标准答案，评估“从分离人声自动提取乐谱+音高线”的准确度。

    python compare_reference.py <analysis_snapshot.json> <ustx> [--part 0]

1. 由 HachiShifter GAME+FCPE 分析快照拼出逐帧音高线（10 ms）
2. 由 USTX 拼出逐帧乐谱音高；在 ±20 s 内搜索时间偏移，使两者吻合帧数最多
3. 指标：
   - 音高线：乐谱有音的帧中，提取到音高的比例；其中与乐谱音高相差 ≤50 cents / 八度错误的比例
   - 逐音符：每个 USTX 音符内音高线中位数 vs 乐谱音高
   - 音符切分：GAME 音符起点与 USTX 起点匹配（±60 ms 且音高相同）的精确率/召回率
"""
import argparse
import json

import numpy as np
import yaml

HOP = 0.01


def f0_track(snap, total):
    f0 = np.full(int(total / HOP) + 1, np.nan)
    notes = []
    for tr in snap["tracks"]:
        for c in tr["clips"]:
            off = c.get("start_seconds", 0.0)
            for n in c["notes"]:
                if n.get("native_unpitched"):
                    continue
                s = off + n["start_seconds"]
                notes.append((s, n["duration_seconds"], n["midi"]))
                for p in n.get("contour", []):
                    if p.get("voiced"):
                        k = int(round((s + p["time_seconds"]) / HOP))
                        if 0 <= k < len(f0):
                            f0[k] = n["midi"] + p["relative_cents"] / 100.0
    # 等距插补 5 ms 轮廓到 10 ms 帧时可能留洞：短洞线性补齐
    idx = np.where(~np.isnan(f0))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 3:
            f0[a:b + 1] = np.linspace(f0[a], f0[b], b - a + 1)
    return f0, sorted(notes)


def score_track(ustx, part, total):
    d = yaml.safe_load(open(ustx, encoding="utf-8-sig"))
    bpm, res = float(d["bpm"]), int(d.get("resolution", 480))
    p = d["voice_parts"][part]
    sec = lambda t: (p["position"] + t) / res * 60 / bpm  # noqa: E731
    roll = np.full(int(total / HOP) + 1, np.nan)
    notes = []
    for n in p["notes"]:
        s, e = sec(n["position"]), sec(n["position"] + n["duration"])
        notes.append((s, e - s, int(n["tone"])))
        roll[int(s / HOP):int(e / HOP)] = n["tone"]
    return roll, sorted(notes)


def shifted(arr, lag):
    out = np.full_like(arr, np.nan)
    if lag >= 0:
        out[lag:] = arr[:len(arr) - lag]
    else:
        out[:lag] = arr[-lag:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("ustx")
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()
    snap = json.load(open(a.snapshot, encoding="utf-8"))
    total = 400.0
    f0, gnotes = f0_track(snap, total)
    roll0, unotes = score_track(a.ustx, a.part, total)

    best = (0, -1)
    for lag in range(-2000, 2001, 2):  # ±20 s，USTX 时间 + lag = 音频时间
        r = shifted(roll0, lag)
        both = ~np.isnan(r) & ~np.isnan(f0)
        hit = int(np.sum(np.abs(f0[both] - r[both]) <= 0.5))
        if hit > best[1]:
            best = (lag, hit)
    lag = best[0]
    for fine in range(lag - 2, lag + 3):
        r = shifted(roll0, fine)
        both = ~np.isnan(r) & ~np.isnan(f0)
        hit = int(np.sum(np.abs(f0[both] - r[both]) <= 0.5))
        if hit > best[1]:
            best = (fine, hit)
    lag = best[0]
    off = lag * HOP
    roll = shifted(roll0, lag)
    print(f"时间偏移：音频时间 = USTX 时间 {off:+.2f} s")

    has = ~np.isnan(roll)
    got = has & ~np.isnan(f0)
    d = np.abs(f0[got] - roll[got])
    print(f"\n【音高线】乐谱有音的帧 {has.sum()}，提取到音高 {got.sum() / has.sum():.0%}")
    print(f"  其中与乐谱相差 ≤50 cents {np.mean(d <= 0.5):.0%}，≤100 cents {np.mean(d <= 1.0):.0%}，"
          f"八度错误(11~13 半音) {np.mean((d >= 11) & (d <= 13)):.1%}")
    extra = (~has) & ~np.isnan(f0)
    print(f"  乐谱休止处却提取到音高的帧 {extra.sum()}（占提取帧 {extra.sum() / max(1, (~np.isnan(f0)).sum()):.0%}，多为和声/伴奏残留或尾音）")

    per = []
    for s, dur, tone in unotes:
        k0, k1 = int((s + off + 0.03) / HOP), int((s + off + dur - 0.02) / HOP)
        seg = f0[k0:max(k0 + 1, k1)]
        seg = seg[~np.isnan(seg)]
        per.append(np.nan if len(seg) < 2 else float(np.median(seg)) - tone)
    per = np.array(per)
    ok = ~np.isnan(per)
    print(f"\n【逐音符】USTX 音符 {len(per)}，测到音高 {ok.mean():.0%}；中位数与乐谱相差 ≤50 cents {np.mean(np.abs(per[ok]) <= 0.5):.0%}，"
          f"差 1 个半音以上 {np.mean(np.abs(per[ok]) > 1):.0%}")

    tp = 0
    used = set()
    for s, dur, tone in unotes:
        for j, (gs, gd, gt) in enumerate(gnotes):
            if j not in used and abs(gs - (s + off)) <= 0.06 and round(gt) == tone:
                tp += 1
                used.add(j)
                break
    in_range = [g for g in gnotes if min(u[0] for u in unotes) + off - 1 <= g[0] <= max(u[0] for u in unotes) + off + 1]
    print(f"\n【音符切分 GAME】USTX {len(unotes)} 个，GAME 在同一范围 {len(in_range)} 个；起点±60ms 且音高相同的匹配 {tp}")
    print(f"  召回率 {tp / len(unotes):.0%}，精确率 {tp / max(1, len(in_range)):.0%}")
    json.dump({"offset_seconds": off}, open(a.snapshot.replace("analysis_snapshot.json", "alignment.json"), "w"))


if __name__ == "__main__":
    main()
