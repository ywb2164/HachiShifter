"""把一条轨的音符轮流铺到 K 条轨上（第 i 个音放到第 i % K 条轨），各轨音符互不相邻，播放时仍然连续。

不延长音尾、不加包络，只做拆轨。用法（库函数）：split_round_robin(mcp, track_id, settings, k) -> [轨道 id, ...]
"""
from __future__ import annotations

from legato import notes_of


def split_round_robin(mcp, track_id: str, settings: dict, k: int) -> list[str]:
    tr, clip, _ = notes_of(mcp, track_id)
    base = tr.get("name") or "track"
    tracks = [track_id]
    for j in range(1, k):
        before = {t["id"] for t in mcp.call_json("project_snapshot")["tracks"]}
        ok, text = mcp.call("add_track", name=f"{base}_{j + 1}")
        if not ok:
            raise RuntimeError("add_track: " + text)
        new = next(t["id"] for t in mcp.call_json("project_snapshot")["tracks"] if t["id"] not in before)
        mcp.call("set_track", track_id=new, **settings)
        ok, text = mcp.call("duplicate_clip", clip_id=clip["id"], start_seconds=clip.get("start_seconds", 0.0), track_id=new)
        if not ok:
            raise RuntimeError("duplicate_clip: " + text)
        tracks.append(new)
    for j, t in enumerate(tracks):
        _, _, notes = notes_of(mcp, t)
        sung = [n for n in notes if n.get("label", "").upper() != "R"]
        for i, n in enumerate(sung):
            if i % k != j:
                mcp.call("remove_note", note_id=n["id"])
    return tracks
