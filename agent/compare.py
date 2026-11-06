"""对照实验：多个 Skill 版本 × 多个任务 × 重复次数，汇总成表。

    D:/miniconda/envs/otomad/python.exe compare.py --skills skill/SKILL_v1.md skill/SKILL.md --repeat 2
结果写到 runs/compare_<时间>.json，并打印表格。
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="*", default=sorted(glob.glob(str(HERE / "tasks" / "*.json"))))
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--model", default="deepseek-flash")
    args = ap.parse_args()
    rows = []
    for task in args.tasks:
        for skill in args.skills:
            for r in range(args.repeat):
                before = set(glob.glob(str(HERE / "runs" / "*")))
                subprocess.run([sys.executable, str(HERE / "runner.py"), task, "--skill", skill, "--model", args.model],
                               cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                new = sorted(set(glob.glob(str(HERE / "runs" / "*"))) - before)
                s = json.loads((Path(new[-1]) / "summary.json").read_text(encoding="utf-8")) if new else {}
                c = s.get("final_check") or {}
                row = {"task": Path(task).stem, "skill": Path(skill).stem, "repeat": r + 1, "run": new[-1] if new else None,
                       "score": c.get("score"), "intelligibility": c.get("intelligibility"),
                       "harmonic": c.get("harmonic_ratio"), "voicing": c.get("voicing_mean"),
                       "renders": s.get("renders"), "steps": s.get("steps"), "cost": s.get("cost_usd_est"),
                       "finished": s.get("finished"), "error": s.get("error"),
                       "restored": "[系统]" in (s.get("agent_summary") or "")}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    out = HERE / "runs" / f"compare_{time.strftime('%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n任务            Skill      分数(各次)        平均   花费")
    for task in sorted({r["task"] for r in rows}):
        for skill in [Path(s).stem for s in args.skills]:
            rs = [r for r in rows if r["task"] == task and r["skill"] == skill]
            sc = [r["score"] for r in rs if r["score"] is not None]
            print(f"{task:15s} {skill:10s} {str(sc):18s} {sum(sc) / max(1, len(sc)):5.1f}  ${sum(r['cost'] or 0 for r in rs):.3f}")
    print("结果:", out)


if __name__ == "__main__":
    main()
