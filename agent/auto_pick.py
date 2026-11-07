"""确定性选字（不经 DeepSeek），用来验证选字规则本身：

    python auto_pick.py tasks_local/t10_wonder_multi.json --name rules_v1 [--no-chains] [--render JSON] [--vowel-fit trim --max-vowel-compression 1.5]

规则（来自人工范例 teio.mpd 的统计）：
1. 每个音取 search_materials 打分第一（打分里下移重罚、上移 ≤+10 基本不罚）
2. 从左到右，能用同一条录音里连续说出的链覆盖 4/3/2 个音时优先用链（链内任何一个音下移不超过 2 半音）
3. 画参考音高线 → 渲染 → 音量平衡 → 再渲染
输出 runs/<task>_<name>/summary.json（picks 格式与 runner 相同，可以 replay_render）。
"""
import argparse
import json
import time
from pathlib import Path

import tools
from tools import Session


def chain_ok(s, cid, max_down=2.0):
    for m in s.cands[cid]["chain"]:
        meta = s.index.get(m["region_id"], {})
        med = meta.get("median_midi")
        if med is not None and s.notes[m["note"]]["midi"] - med < -max_down:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--name", required=True)
    ap.add_argument("--render", default='{"pitch_algorithm":"mld5","stretch_algorithm":"melodyne-hybrid","render_order":"process-then-splice","merge_contiguous_source":true}')
    ap.add_argument("--no-chains", action="store_true")
    ap.add_argument("--vowel-fit", default="trim")
    ap.add_argument("--max-vowel-compression", type=float, default=1.5)
    ap.add_argument("--pitch", default="reference")
    ap.add_argument("--legato", type=float, default=0.0, help=">0 时做两轨交替尾巴（旧做法），默认不做")
    a = ap.parse_args()
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    tools.RENDER.clear(); tools.RENDER.update(json.loads(a.render))
    tools.FIT.clear()
    if a.vowel_fit:
        tools.FIT.update(vowel_fit=a.vowel_fit, max_vowel_compression=a.max_vowel_compression)
    run = Path(__file__).parent / "runs" / f"{task['id']}_{a.name}_{time.strftime('%m%d-%H%M%S')}"
    run.mkdir(parents=True)
    s = Session(task, run)
    n = len(s.notes)
    chosen, chains, covered = {}, [], set()
    if not a.no_chains:
        i = 0
        while i < n:
            took = False
            for L in (4, 3, 2):
                if i + L > n or not all(s.notes[k]["char"] for k in range(i, i + L)):
                    continue
                r = s.search_phrase(i, i + L - 1)
                for c in r.get("chains", []):
                    if c["notes"] == f"{i}-{i + L - 1}" and chain_ok(s, c["chain_id"]):
                        chains.append(c["chain_id"]); covered.update(range(i, i + L)); took = True
                        break
                if took:
                    i += L
                    break
            if not took:
                i += 1
    for k in range(n):
        if k in covered or not s.notes[k]["char"]:
            continue
        r = s.search_materials(k, limit=1)
        if r.get("candidates"):
            chosen[k] = r["candidates"][0]["candidate_id"]
    res = s.apply_materials(choices=[{"note": k, "candidate_id": c} for k, c in chosen.items()], chain_ids=chains)
    print("apply:", res, "chains:", len(chains), "notes in chains:", len(covered))
    print("pitch:", s.apply_pitch_curves(a.pitch) if a.pitch else None)
    r1 = s.render_and_check()
    print("render1:", {k: r1.get(k) for k in ("score", "quality")} if isinstance(r1, dict) else r1)
    print("balance:", {k: v for k, v in s.balance_loudness().items() if k != "largest"})
    if a.legato > 0:
        print("legato:", s.finalize_legato(tail_ratio=a.legato))
    r2 = s.render_and_check()
    picks = {str(nn["index"]): dict(nn["pick"], note=nn["index"], material=nn["material"]) for nn in s.notes if nn.get("pick")}
    shifts = []
    for nn in s.notes:
        if nn.get("pick"):
            med = s.index.get(nn["pick"]["region_id"], {}).get("median_midi")
            if med is not None:
                shifts.append(nn["midi"] - med)
    summary = {"picks": picks, "pitch_source": a.pitch, "chains": len(chains), "notes_in_chains": len(covered),
               "final": r2, "render": tools.RENDER, "fit": tools.FIT,
               "shift_median": float(sorted(shifts)[len(shifts) // 2]) if shifts else None,
               "down3_frac": round(sum(x <= -3 for x in shifts) / max(1, len(shifts)), 2)}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("final:", {k: r2.get(k) for k in ("score", "in_tune_ratio", "harmonic_ratio", "voicing_mean")} if isinstance(r2, dict) else r2)
    print("shift median", summary["shift_median"], "down>=3st frac", summary["down3_frac"], "run", run)
    s.close()


if __name__ == "__main__":
    main()
