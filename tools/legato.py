"""人工式衔接：两轨交替 + 音尾延长淡出 + 压音头。

HachiShifter 原生轨同一片段内前一个音会在后一个音开始处被截断，所以：
1. 复制片段到第二条轨，A 轨保留偶数音、B 轨保留奇数音（每轨各自单声部）
2. 后面紧挨着有音的音符：结尾延长 tail_ms，从下一个音的辅音开始处淡出到 -40 dB（叠在下一个音头上）
3. 可选：音头（辅音到音符起点后 40 ms）比元音主体响 > head_db 的，按超出量压低

用法（库函数）：apply_legato(mcp, track_id, tail_ms=60, head_rules=None)
"""
from __future__ import annotations


def notes_of(mcp, track_id):
    tr = next(t for t in mcp.call_json("project_snapshot")["tracks"] if t["id"] == track_id)
    clip = tr["clips"][0]
    notes = sorted([n for c in tr["clips"] for n in c["notes"]], key=lambda n: n["start_seconds"])
    return tr, clip, notes


def split_alternate(mcp, track_id, settings: dict) -> tuple[str, str]:
    """把一条轨拆成交替的两条轨，返回 (A 轨, B 轨)。"""
    tr, clip, notes = notes_of(mcp, track_id)
    ok, text = mcp.call("add_track", name=(tr.get("name") or "track") + "_B")
    b_track = next(t for t in mcp.call_json("project_snapshot")["tracks"] if t["id"] not in {track_id} and t.get("name", "").endswith("_B"))["id"]
    mcp.call("set_track", track_id=b_track, **settings)
    ok, text = mcp.call("duplicate_clip", clip_id=clip["id"], start_seconds=clip.get("start_seconds", 0.0), track_id=b_track)
    if not ok:
        raise RuntimeError("duplicate_clip: " + text)
    _, _, a_notes = notes_of(mcp, track_id)
    _, _, b_notes = notes_of(mcp, b_track)
    sung_a = [n for n in a_notes if n.get("label", "").upper() != "R"]
    sung_b = [n for n in b_notes if n.get("label", "").upper() != "R"]
    for i, (na, nb) in enumerate(zip(sung_a, sung_b)):
        mcp.call("remove_note", note_id=(na if i % 2 else nb)["id"])
    return track_id, b_track


def apply_legato(mcp, a_track: str, b_track: str, tail_ms: float = 60.0, fade_floor_db: float = -40.0,
                 head_cut: dict[int, float] | None = None) -> dict:
    """在交替的两轨上延长音尾并淡出；head_cut = {全局音符序号: 压低 dB}。"""
    _, a_clip, a_notes = notes_of(mcp, a_track)
    _, b_clip, b_notes = notes_of(mcp, b_track)
    tagged = [(a_clip.get("start_seconds", 0.0) + n["start_seconds"], n, a_clip) for n in a_notes if n.get("label", "").upper() != "R"]
    tagged += [(b_clip.get("start_seconds", 0.0) + n["start_seconds"], n, b_clip) for n in b_notes if n.get("label", "").upper() != "R"]
    tagged.sort(key=lambda x: x[0])
    extended = cut = 0
    for k, (start, n, clip) in enumerate(tagged):
        env = []
        if head_cut and head_cut.get(k):
            g = -abs(head_cut[k])
            cons = float(n.get("consonant_seconds") or 0.0)
            env += [[-max(cons, 0.005), g], [0.04, g], [0.09, 0.0]]
            cut += 1
        dur = n["duration_seconds"]
        if k + 1 < len(tagged):
            nstart, nxt, _ = tagged[k + 1]
            gap = nstart - (start + dur)
            if abs(gap) < 0.03:
                ext = min(tail_ms / 1000.0, 0.5 * nxt["duration_seconds"])
                ncons = float(nxt.get("consonant_seconds") or 0.03)
                fade_from = max(0.5 * dur, (nstart - ncons) - start)
                new_dur = dur + ext
                mcp.call("resize_note", note_id=n["id"], start_seconds=n["start_seconds"], duration_seconds=new_dur)
                env += [[fade_from, 0.0], [new_dur, fade_floor_db]]
                extended += 1
        if env:
            env.sort(key=lambda p: p[0])
            if env[0][0] > 0:
                env.insert(0, [0.0, 0.0])
            mcp.call("set_note", note_id=n["id"], amplitude_envelope=env)
    return {"notes": len(tagged), "extended": extended, "head_cut": cut}
