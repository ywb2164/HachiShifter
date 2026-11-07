"""Melodyne 式音高处理：不画逐点曲线，只用 音高中心 + 部分拉平（漂移/调制）+ 必要时切开，去大致贴旋律线。

HachiShifter 渲染音高 = 中心 + drift×素材慢变偏离 + modulation×素材颤音（modulation=0 时完全拉平）。
对每个音：
1. 在 (drift, modulation) 网格上，中心取“目标线 − 素材自然走势”的中位数，误差 = 平均绝对偏差（音分）；
   拉平有代价 lam×((1−d)+(1−m))，素材本身走势能贴住旋律就不拉平（人工范例 76/81 个音保留原走势）。
2. 切分看音高线本身的复杂度，不追求刀数少：对每一段，若它对“平滑后的目标线”的平均偏差 > tol，
   且在某处切开能让这一段偏差下降 ≥ min_gain_frac、两边中心相差 ≥ min_step，就切，并对两半继续判断。
   唯一硬限制是每段 ≥ min_piece（Melodyne 里常见的 30 ms 小音）。短而平稳的音自然不切，
   长音里的复杂转音会切很多刀。目标线先做 smooth 秒的滑动平均：颤音之类的快速来回不切成锯齿，
   交给调制保留素材自己的颤动；台阶式转音、慢滑音才切。
"""
from __future__ import annotations

import numpy as np

DRIFTS = (1.0, 0.75, 0.5, 0.25, 0.0)
MODS = (1.0, 0.6, 0.3, 0.1)


def _natural(contour: list[dict], d: float, m: float, t: np.ndarray):
    if not contour:
        return np.zeros_like(t), np.zeros_like(t, dtype=bool)
    ct = np.array([p["time_seconds"] for p in contour])
    rel = np.array([p["relative_cents"] for p in contour])
    wv = np.array([p.get("without_vibrato_cents", p["relative_cents"]) for p in contour])
    voiced = np.array([bool(p["voiced"]) for p in contour])
    dev = d * wv + m * (rel - wv)
    vi = np.interp(t, ct, dev) if voiced.any() else np.zeros_like(t)
    near = np.clip(np.searchsorted(ct, t), 0, len(ct) - 1)
    return vi, voiced[near]


def _target(R, t):
    rt = np.array([p[0] for p in R]); rv = np.array([p[1] for p in R]) * 100.0
    return np.interp(t, rt, rv)


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(x) < 3:
        return x
    k = np.ones(width) / width
    pad = width // 2
    return np.convolve(np.pad(x, (pad, width - 1 - pad), mode="edge"), k, mode="valid")


def fit_note(contour, duration, R, grid=0.005, lam=12.0, tol=30.0, min_gain_frac=0.35, min_step=35.0,
             min_piece=0.03, smooth=0.06, penalty=600.0, max_splits=None, **_legacy):
    """R: [(t 秒, 目标 midi)]，t 相对音符起点。max_splits=0 表示不切；None 表示不限（只受 min_piece 限制）。
    返回 {drift, modulation, center_midi, splits:[(t, center_midi)], err_cents, err_unsplit_cents}"""
    t = np.arange(0.0, max(duration, grid), grid)
    goal = _target(R, t)
    best = None
    for d in DRIFTS:
        for m in MODS:
            nat, v = _natural(contour, d, m, t)
            if v.sum() < 3:
                v = np.ones_like(t, dtype=bool)
            c = float(np.median(goal[v] - nat[v]))
            err = float(np.mean(np.abs(goal[v] - c - nat[v])))
            cost = err + lam * ((1 - d) + (1 - m))
            if best is None or cost < best[0]:
                best = (cost, d, m, c, err, nat, v)
    _, d, m, c, err, nat, v = best
    # 切分只看慢变部分：平滑后的目标线减素材自然走势
    slow = _smooth(goal, max(1, int(round(smooth / grid))))
    res = np.where(v, slow - c - nat, np.nan)
    minw = max(1, int(round(min_piece / grid)))

    def stats(a, b):
        r = res[a:b]; r = r[np.isfinite(r)]
        if len(r) < 3:
            return None
        med = float(np.median(r))
        return med, float(np.mean(np.abs(r - med))), len(r)

    cuts: list[int] = []
    total = stats(0, len(t))
    if max_splits != 0 and total is not None and total[1] > tol and len(t) >= 2 * minw:
        # 最优分段（动态规划）：总偏差（音分·帧）+ 每多一段 penalty；中间凸起的台阶也能一次切出来
        step = 2
        idx = list(range(0, len(t) - minw + 1, step)) + [len(t)]
        idx = sorted(set(idx))
        cost_cache = {}

        def cost(i, j):
            key = (i, j)
            if key not in cost_cache:
                s_ = stats(i, j)
                cost_cache[key] = 0.0 if s_ is None else s_[1] * s_[2]
            return cost_cache[key]

        best_cost = {0: 0.0}; back = {0: None}
        for j in idx[1:]:
            cand = None
            for i in idx:
                if i >= j:
                    break
                if i not in best_cost or (j - i < minw) or (i != 0 and i < minw) or (j != len(t) and len(t) - j < minw):
                    continue
                val = best_cost[i] + cost(i, j) + (penalty if i > 0 else 0.0)
                if cand is None or val < cand[0]:
                    cand = (val, i)
            if cand:
                best_cost[j], back[j] = cand
        j = len(t)
        while j in back and back[j] is not None and back[j] > 0:
            cuts.append(back[j]); j = back[j]
        cuts.sort()
        # 相邻两段中心差不到 min_step 的合并（不是真台阶）
        changed = True
        while changed and cuts:
            changed = False
            bnd = [0] + cuts + [len(t)]
            meds = [(stats(x, y) or (0.0,))[0] for x, y in zip(bnd, bnd[1:])]
            diffs = [abs(meds[i + 1] - meds[i]) for i in range(len(meds) - 1)]
            k = int(np.argmin(diffs))
            if diffs[k] < min_step:
                cuts.pop(k); changed = True
        if max_splits is not None and len(cuts) > max_splits:
            cuts = cuts[:max_splits]
    cuts.sort()
    bounds = [0] + cuts + [len(t)]
    centers = []
    for a, b in zip(bounds, bounds[1:]):
        s = stats(a, b)
        centers.append((c + (s[0] if s else 0.0)) / 100.0)
    fitted = np.concatenate([np.full(b - a, cm * 100.0) for (a, b), cm in zip(zip(bounds, bounds[1:]), centers)]) + nat
    err_after = float(np.mean(np.abs((goal - fitted)[v])))
    splits = [(round(float(t[k]), 4), round(cm, 3)) for k, cm in zip(cuts, centers[1:])]
    return {"drift": d, "modulation": m, "center_midi": round(centers[0], 3), "splits": splits,
            "err_cents": round(err_after, 1), "err_unsplit_cents": round(err, 1)}
