"""DeepSeek 调音 Agent 执行器。

    D:/miniconda/envs/otomad/python.exe runner.py tasks/t01_dajiahao.json [--model deepseek-flash] [--max-steps 40]

- Skill = skill/SKILL.md（作为 system prompt；版本号取文件内容哈希，写进结果）
- 每次运行一个目录 runs/<任务>_<时间>/：trace.jsonl（每条消息/工具调用）、render_*.wav、result.hjpx、summary.json
- 思考模式下必须把 reasoning_content 原样传回（DeepSeek 文档要求，否则 400）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from openai import OpenAI

from tools import TOOL_SPECS, Session

HERE = Path(__file__).resolve().parent
PRICE = {"deepseek-flash": (0.3, 0.006, 1.2), "deepseek-v4-pro": (1.32, 0.044, 3.96)}  # 每百万 token 美元（峰时，偏保守）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--skill", default=str(HERE / "skill" / "SKILL.md"))
    ap.add_argument("--effort", default="high")
    args = ap.parse_args()

    task = json.loads(Path(args.task).read_text(encoding="utf-8"))
    skill = Path(args.skill).read_text(encoding="utf-8")
    skill_ver = hashlib.sha1(skill.encode("utf-8")).hexdigest()[:8]
    run_dir = HERE / "runs" / f"{task['id']}_{time.strftime('%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    trace = (run_dir / "trace.jsonl").open("w", encoding="utf-8")

    def log(kind, **kw):
        trace.write(json.dumps({"t": round(time.time(), 2), "kind": kind, **kw}, ensure_ascii=False) + "\n")
        trace.flush()

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    session = Session(task, run_dir)
    tools = [{"type": "function", "function": s} for s in TOOL_SPECS]
    messages = [{"role": "system", "content": skill},
                {"role": "user", "content": f"新的调音任务：{task['title']}。请按 Skill 的流程完成，最后调用 finish。"}]
    log("start", task=task["id"], model=args.model, skill_version=skill_ver)
    usage = {"prompt": 0, "cached": 0, "completion": 0}
    t0 = time.time()
    nudges = 0
    steps = 0
    error = None
    try:
        while steps < args.max_steps and not session.finished:
            steps += 1
            resp = client.chat.completions.create(
                model=args.model, messages=messages, tools=tools, reasoning_effort=args.effort,
                extra_body={"thinking": {"type": "enabled"}})
            u = resp.usage
            usage["prompt"] += u.prompt_tokens
            usage["completion"] += u.completion_tokens
            usage["cached"] += getattr(u, "prompt_cache_hit_tokens", 0) or 0
            msg = resp.choices[0].message
            extra = msg.model_extra or {}
            reasoning = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content")
            assistant = {"role": "assistant", "content": msg.content or ""}
            if reasoning is not None:
                assistant["reasoning_content"] = reasoning
            if msg.tool_calls:
                assistant["tool_calls"] = [{"id": tc.id, "type": "function",
                                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                           for tc in msg.tool_calls]
            messages.append(assistant)
            log("assistant", step=steps, content=assistant["content"], reasoning=reasoning,
                tool_calls=assistant.get("tool_calls"))
            print(f"[{steps}] {(assistant['content'] or '')[:120]}")
            if not msg.tool_calls:
                if nudges >= 2:
                    break
                nudges += 1
                messages.append({"role": "user", "content": "请继续用工具完成任务；全部完成后调用 finish。"})
                continue
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    a = json.loads(tc.function.arguments or "{}")
                    result = getattr(session, name)(**a) if name in {s["name"] for s in TOOL_SPECS} else {"error": f"没有这个工具: {name}"}
                except Exception as e:  # 工具出错也要回给模型，让它自己调整
                    result = {"error": f"{type(e).__name__}: {e}"}
                text = json.dumps(result, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
                log("tool", step=steps, name=name, args=tc.function.arguments, result=result)
                print(f"    → {name}({(tc.function.arguments or '')[:80]}) {text[:150]}")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        log("error", error=error)
        print("错误:", error)
    finally:
        session.close()
    price = PRICE.get(args.model, (0, 0, 0))
    cost = ((usage["prompt"] - usage["cached"]) * price[0] + usage["cached"] * price[1] + usage["completion"] * price[2]) / 1e6
    summary = {"task": task["id"], "model": args.model, "skill_version": skill_ver, "steps": steps,
               "finished": session.finished, "error": error, "seconds": round(time.time() - t0),
               "tokens": usage, "cost_usd_est": round(cost, 4), "renders": session.renders,
               "final_check": session.last_check, "agent_summary": session.final_summary,
               "notes": session._notes_view(),
               "picks": session.final_picks or {n["index"]: dict(n["pick"], material=n["material"]) for n in session.notes if n.get("pick")},
               "pitch_source": session.pitch_source}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log("end", **{k: v for k, v in summary.items() if k != "notes"})
    trace.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "notes"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
