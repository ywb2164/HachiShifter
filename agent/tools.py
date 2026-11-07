"""DeepSeek 调音 Agent 的工具层：一个 Session 持有 HachiShifter 工程、素材索引和检查器。

DeepSeek 只看到这里暴露的 7 个工具（见 TOOL_SPECS）；数值计算（候选打分、对齐、拉平、渲染、检查）都在代码里。
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import otomad_config as cfg  # noqa: E402

import difflib
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pypinyin import Style, lazy_pinyin

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT / "tools"))
from mcp_client import HachiMcp  # noqa: E402
sys.path.insert(0, str(AGENT / "annotator"))
from kana import candidates as kana_candidates, to_romaji  # noqa: E402
from legato import apply_legato, split_alternate  # noqa: E402
from pitchfit import fit_note  # noqa: E402

MIRROR = AGENT / "library"
EXE = str(cfg.EXE)
MODELS = cfg.MODELS
RENDER = {"pitch_algorithm": "world"}  # 盲听结论：world + 完全拉平
FLAT = {"drift": 0, "modulation": 0}
FIT = {}  # replace_note_material vowel fit, e.g. {"vowel_fit": "trim"}: MLD5 needs trim (no 10x vowel squeeze)

_ASR = None


def asr_model():
    global _ASR
    if _ASR is None:
        from funasr import AutoModel
        _ASR = AutoModel(model="paraformer-zh", disable_update=True)
    return _ASR


def quality_metrics(y: np.ndarray, sr: int, ref: dict | None = None) -> dict:
    """整段渲染的音质指标（只用你 7 个评分做过方向检查，权重没有拟合）。

    ref = 素材库自身的基线 {"harmonic_ratio", "voicing_mean"}：给了就按“相对素材本身”打分
    （和素材一样纯净 = 1，只剩一半 = 0），没给就用中文语音素材上定的绝对标准。"""
    clipped = float(np.mean(np.abs(y) > 0.98)) * 1000
    H, _ = librosa.effects.hpss(y)
    harm = float(np.sum(H ** 2) / (np.sum(y ** 2) + 1e-12))
    _, _, vp = librosa.pyin(y, fmin=65, fmax=1500, sr=sr, hop_length=220)
    rms = librosa.feature.rms(y=y, hop_length=220)[0]
    loud = rms > rms.max() * 0.1
    n = min(len(vp), len(loud))
    vmean = float(np.nanmean(vp[:n][loud[:n]])) if loud[:n].any() else 0.0
    if ref:
        hn = float(np.clip((harm / max(ref["harmonic_ratio"], 1e-3) - 0.5) / 0.5, 0, 1))
        vn = float(np.clip((vmean / max(ref["voicing_mean"], 1e-3) - 0.5) / 0.5, 0, 1))
    else:
        hn = float(np.clip((harm - 0.5) / 0.45, 0, 1))
        vn = float(np.clip((vmean - 0.45) / 0.35, 0, 1))
    return {"clipped_permille": round(clipped, 2), "harmonic_ratio": round(harm, 2), "voicing_mean": round(vmean, 2),
            "harmonic_norm": hn, "voicing_norm": vn}


def library_reference(folder: Path, sample: int = 30) -> dict:
    """素材库自身的谐波占比 / 有声概率基线（抽样计算，缓存到 quality_ref.json）。"""
    cache = folder / "quality_ref.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    files = sorted(folder.rglob("*.wav"))
    files = files[:: max(1, len(files) // sample)][:sample]
    hs, vs = [], []
    for f in files:
        y, sr = librosa.load(f, sr=22050, mono=True)
        q = quality_metrics(y, sr)
        hs.append(q["harmonic_ratio"])
        vs.append(q["voicing_mean"])
    ref = {"harmonic_ratio": round(float(np.mean(hs)), 3), "voicing_mean": round(float(np.mean(vs)), 3), "files": len(files)}
    cache.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    return ref


def han(s: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", s))


class Session:
    def __init__(self, task: dict, run_dir: Path):
        self.task = task
        self.run_dir = run_dir
        self.roots = [str(MIRROR / r) for r in task["roots"]]
        self.index = {}
        for r in task["roots"]:
            f = MIRROR / r / "annotations.json"
            for row in json.loads(f.read_text(encoding="utf-8"))["rows"]:
                self.index[row["region_id"]] = {**row, "folder": r}
        self.mcp = HachiMcp(EXE, [f"--roots={MIRROR}"])
        self.cands: dict[str, dict] = {}
        self.lang = task.get("language", "zh")
        self.legato_done = False
        self.final_picks = None
        self.pitch_source = None
        self.pitch_mode = "curve"
        self._qcache: dict = {}
        # 日语歌声切片等“本身就不纯净”的素材库：音质按相对素材自身基线打分
        self.qref = None
        if self.lang == "ja":
            refs = [library_reference(MIRROR / r) for r in task["roots"]]
            self.qref = {k: float(np.mean([r[k] for r in refs])) for k in ("harmonic_ratio", "voicing_mean")}
        self.renders = 0
        self.finished = False
        self.final_summary = ""
        self.last_check = None
        self.volume = 1.0
        self.best = None  # {"score":..., "picks": {note: pick}, "render": ...}
        self._build_project()

    # ---------- 工程 ----------
    def _build_project(self) -> None:
        bpm = self.task["bpm"]
        ust = self.run_dir / "score.ust"
        lines = ["[#VERSION]", "UST Version1.2", "[#SETTING]", f"Tempo={bpm:.2f}", "Tracks=1",
                 f"ProjectName={self.task['id']}", "VoiceDir=", "Mode2=True"]
        k = 0
        pos = 0.0
        for n in self.task["melody"]:
            if n["start_beat"] > pos + 1e-6:  # 休止符
                lines += [f"[#{k:04d}]", f"Length={round((n['start_beat'] - pos) * 480)}", "Lyric=R", "NoteNum=60"]
                k += 1
            lines += [f"[#{k:04d}]", f"Length={round(n['beats'] * 480)}", "Lyric=a", f"NoteNum={n['midi']}", "Modulation=0"]
            k += 1
            pos = n["start_beat"] + n["beats"]
        lines.append("[#TRACKEND]")
        ust.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
        self.mcp.call("project_new")
        ok, text = self.mcp.call("import_ust", path=str(ust))
        if not ok:
            raise RuntimeError("import_ust: " + text)
        snap = self.mcp.call_json("project_snapshot")
        self.track_id = snap["tracks"][-1]["id"]
        ok, text = self.mcp.call("set_track", track_id=self.track_id, **RENDER)
        notes = sorted((n for c in snap["tracks"][-1]["clips"] for n in c["notes"]), key=lambda n: n["start_seconds"])
        notes = [n for n in notes if n.get("label", "").upper() != "R"]
        sec_per_beat = 60.0 / bpm
        self.notes = []
        for i, (n, m) in enumerate(zip(notes, self.task["melody"])):
            self.notes.append({"index": i, "id": n["id"], "start": m["start_beat"] * sec_per_beat,
                               "duration": m["beats"] * sec_per_beat, "midi": m["midi"],
                               "char": None, "pinyin": None, "material": None})
        for n, m in zip(self.notes, self.task["melody"]):
            if m.get("lyric"):
                n["char"] = m["lyric"]
                n["pinyin"] = to_romaji(m["lyric"]) if self.lang == "ja" else None
                self.mcp.call("set_note", note_id=n["id"], label=m["lyric"])
        if self.lang != "ja" and any(n["char"] for n in self.notes):
            seq = [n for n in self.notes if n["char"]]
            for n, py in zip(seq, lazy_pinyin("".join(n["char"] for n in seq), style=Style.NORMAL)):
                n["pinyin"] = py

    def _notes_view(self):
        return [{"note": n["index"], "start_s": round(n["start"], 3), "dur_ms": round(n["duration"] * 1000),
                 "midi": n["midi"], "pitch": librosa.midi_to_note(n["midi"], unicode=False),
                 "lyric": n["char"], "material": n["material"]} for n in self.notes]

    # ---------- 工具 ----------
    def get_task(self):
        return {"title": self.task["title"], "bpm": self.task["bpm"], "lyrics": self.task["lyrics"],
                "language": self.lang, "lyrics_prefilled": all(n["char"] for n in self.notes),
                "pitch_curve_sources": list(self.task.get("pitch_curves", {})) + ["flat"],
                "brief": self.task.get("brief", ""), "note_count": len(self.notes),
                "material_folders": self.task["roots"], "notes": self._notes_view()}

    def assign_lyrics(self, assignments: list):
        chars = {}
        for a in assignments:
            i = int(a["note"])
            if not 0 <= i < len(self.notes):
                return {"error": f"音符编号越界: {i}"}
            if self.lang == "ja":
                ch = str(a["char"]).strip()
                if not to_romaji(ch):
                    return {"error": f"无法识别的假名: {a['char']!r}"}
            else:
                ch = han(str(a["char"]))
                if len(ch) != 1:
                    return {"error": f"每个音符只能放一个汉字，收到 {a['char']!r}"}
            chars[i] = ch
        for i, ch in chars.items():
            self.notes[i]["char"] = ch
            self.notes[i]["material"] = None
        # 按整句上下文转拼音（处理多音字）
        seq = [n for n in self.notes if n["char"]]
        pys = ([to_romaji(n["char"]) for n in seq] if self.lang == "ja"
               else lazy_pinyin("".join(n["char"] for n in seq), style=Style.NORMAL))
        for n, py in zip(seq, pys):
            n["pinyin"] = py
            self.mcp.call("set_note", note_id=n["id"], label=n["char"])
        missing = [n["index"] for n in self.notes if not n["char"]]
        return {"ok": True, "assigned": len(seq), "notes_without_lyric": missing,
                "lyric_line": "".join(n["char"] or "＿" for n in self.notes)}

    def _query(self, lyrics: list[str], sequences=False, limit=60):
        return self.mcp.call_json("material_query", roots=self.roots, lyrics=lyrics, limit=limit,
                                  include_sequences=sequences)

    def search_materials(self, note: int, limit: int = 8):
        n = self.notes[int(note)]
        if not n["char"]:
            return {"error": "这个音符还没有分配歌词，先调用 assign_lyrics"}
        if self.lang == "ja":
            terms = [(r, "同音") for r in kana_candidates(n["pinyin"])] + [(n["char"], "同音")]
        else:
            terms = [(n["char"], "同字"), (n["pinyin"], "同音")]
        key = tuple(t for t, _ in terms)
        if key not in self._qcache:
            self._qcache[key] = self._query(list(key))
        q = self._qcache[key]
        seen, out = set(), []
        for (_, kind), item in zip(terms, q["lyrics"]):
            for c in item["candidates"]:
                if c["region_id"] in seen:
                    continue
                seen.add(c["region_id"])
                meta = self.index.get(c["region_id"], {})
                if meta.get("variant") == "short_consonant" and not self.task.get("allow_short_consonant", False):
                    continue  # 人工盲听：短辅音版本让衔接变差，默认不用
                cons = (c["alignment_seconds"] - c["region_start_seconds"]) * 1000
                vowel = (c["region_end_seconds"] - c["alignment_seconds"]) * 1000
                need = max(60.0, n["duration"] * 1000 - 20)
                stretch = need / max(vowel, 1.0)
                shift = None if meta.get("median_midi") is None else round(n["midi"] - meta["median_midi"], 1)
                score = 1.0 * meta.get("confidence", 0.5)
                score += 0.3 if kind == "同字" else 0.0
                score -= 0.12 * max(0.0, np.log2(max(stretch, 1e-3)) - 1.0)  # 拉伸超过 2 倍开始扣分（盲听：拉太长电音重）
                score -= 0.15 * max(0.0, 0.5 - stretch)  # 素材比音符长很多也扣一点
                if shift is not None:
                    if self.lang == "ja":
                        # 人工范例 teio：90% 的音往上移（中位 +5），下移 ≥3 半音只有 2%；
                        # MLD5 下移越多越虚越电。往下重罚，往上到 +10 基本不罚。
                        score -= 0.10 * max(0.0, -shift - 1) + 0.03 * max(0.0, shift - 10)
                    else:
                        score -= 0.04 * max(0.0, abs(shift) - 5)  # 移调超过 5 个半音开始扣分
                robust = meta.get("robust")
                if robust is not None:
                    score += 0.4 * (robust - 0.5)  # 初审：处理后仍干净的素材优先
                cons_ratio = cons / max(1.0, n["duration"] * 1000)
                score -= 0.6 * max(0.0, cons_ratio - 0.5)  # 辅音超过音符一半：元音被挤掉
                cid = f"n{n['index']}_c{len(self.cands)}"
                self.cands[cid] = {"note": n["index"], "audio_path": c["audio_path"], "audio_id": c["audio_id"],
                                   "region_id": c["region_id"], "alias": c["alias"]}
                out.append({"candidate_id": cid, "match": kind, "source": f"{meta.get('folder', '?')}/{meta.get('name_in_library', Path(c['audio_path']).name)}",
                            "said_in_source": meta.get("asr_text") or "(单字素材)", "confidence": meta.get("confidence"),
                            "version": meta.get("version"),
                            "robust": robust, "short_consonant": meta.get("variant") == "short_consonant",
                            "consonant_ms": round(cons), "consonant_ratio": round(cons_ratio, 2),
                            "vowel_ms": round(vowel), "stretch_x": round(stretch, 2),
                            "semitone_shift": shift, "score": round(score, 3)})
        out.sort(key=lambda x: -x["score"])
        return {"note": n["index"], "lyric": n["char"], "pinyin": n["pinyin"], "target_midi": n["midi"],
                "note_ms": round(n["duration"] * 1000), "total_found": len(out), "candidates": out[:int(limit)]}

    def search_materials_batch(self, first_note: int, last_note: int, top: int = 3):
        """一次给一段音符各返回前几名候选（精简字段），省步数。"""
        out = []
        for i in range(int(first_note), int(last_note) + 1):
            r = self.search_materials(i, limit=int(top))
            if "error" in r:
                out.append({"note": i, "error": r["error"]})
                continue
            out.append({"note": i, "lyric": r["lyric"], "romaji": r["pinyin"], "note_ms": r["note_ms"], "target_midi": r["target_midi"],
                        "found": r["total_found"],
                        "top": [{k: c[k] for k in ("candidate_id", "version", "robust", "short_consonant", "confidence", "consonant_ms", "consonant_ratio",
                                                   "vowel_ms", "stretch_x", "semitone_shift", "score")} for c in r["candidates"]]})
        return {"notes": out}

    def search_phrase(self, first_note: int, last_note: int):
        idx = list(range(int(first_note), int(last_note) + 1))
        chars = [self.notes[i]["char"] for i in idx]
        if not all(chars):
            return {"error": "范围内有音符还没分配歌词"}
        q = self._query(chars, sequences=True, limit=5)
        chains = []
        for s in q.get("sequences", []):
            members = s["members"]
            start = idx[s["start_lyric_index"]]
            cid = f"chain{len(self.cands)}"
            self.cands[cid] = {"chain": [{"note": start + k, "audio_path": m["audio_path"], "audio_id": m["audio_id"],
                                          "region_id": m["region_id"], "alias": m["alias"]} for k, m in enumerate(members)]}
            meta = self.index.get(members[0]["region_id"], {})
            chains.append({"chain_id": cid, "notes": f"{start}-{start + len(members) - 1}",
                           "text": "".join(m["alias"] for m in members),
                           "source": f"{meta.get('folder', '?')}/{meta.get('name_in_library', '')}",
                           "said_in_source": meta.get("asr_text", ""),
                           "min_confidence": min(self.index.get(m["region_id"], {}).get("confidence", 0.5) for m in members)})
        chains.sort(key=lambda c: -int(c["notes"].split("-")[1]) + int(c["notes"].split("-")[0]))
        return {"phrase": "".join(chars), "chains": chains[:12],
                "hint": "链表示同一条录音里连续说出的几个字，整段取用最自然" if chains else "没有找到连续说出这些字的录音，逐字选"}

    def apply_materials(self, choices: list | None = None, chain_ids: list | None = None):
        if self.legato_done:
            return {"error": "已经做过 finalize_legato（拆成两轨），不能再换素材；直接 render_and_check 或 finish"}
        picks = {}
        for cid in chain_ids or []:
            c = self.cands.get(cid)
            if not c or "chain" not in c:
                return {"error": f"未知的 chain_id: {cid}"}
            for m in c["chain"]:
                picks[m["note"]] = m
        for ch in choices or []:
            c = self.cands.get(ch["candidate_id"])
            if not c or "chain" in c:
                return {"error": f"未知的 candidate_id: {ch['candidate_id']}"}
            if int(ch["note"]) != c["note"]:
                return {"error": f"{ch['candidate_id']} 是给音符 {c['note']} 搜的，不能用在音符 {ch['note']}"}
            picks[c["note"]] = c
        if not picks:
            return {"error": "没有给出任何选择"}
        order = sorted(picks)
        req = [{"note_id": self.notes[i]["id"], "audio_path": picks[i]["audio_path"], "audio_id": picks[i]["audio_id"],
                "region_id": picks[i]["region_id"], "alias": picks[i]["alias"]} for i in order]
        ok, text = self.mcp.call("replace_note_material", roots=self.roots, choices=req,
                                 game_model_dir=str(MODELS / "game" / "medium"), fcpe_model=str(MODELS / "fcpe" / "fcpe.onnx"), **FIT)
        if not ok:
            return {"error": text[:400]}
        for i in order:
            self.mcp.call("set_note", note_id=self.notes[i]["id"], **FLAT)
            meta = self.index.get(picks[i]["region_id"], {})
            self.notes[i]["pick"] = picks[i]
            self.notes[i]["material"] = f"{meta.get('name_in_library', Path(picks[i]['audio_path']).name)}#{picks[i]['alias']}"
        redrawn = 0
        curves = self._curves(self.pitch_source)
        for i in order:  # 换了素材的音按当前来源重画（没选来源时也按伴奏定音拉平）
            redrawn += self._draw(self.notes[i], self.pitch_source, curves)
        done = sum(1 for n in self.notes if n["material"])
        return {"ok": True, "applied": len(order), "notes_with_material": done, "notes_total": len(self.notes),
                "pitch_redrawn": redrawn}

    def _tune(self) -> float:
        """整体定音偏移（半音），来自伴奏估计，如 -0.12。"""
        return float(self.task.get("tuning_offset_cents", 0.0)) / 100.0

    def _draw(self, n: dict, source: str | None, curves: dict | None) -> bool:
        """给一个音画线；返回是否画了曲线。同时记下 n["target"]（稳定段目标音高，给音准检查用）。"""
        tune = self._tune()
        absolute = source in set(self.task.get("pitch_curves_absolute", ["reference"]))
        pts = (curves or {}).get(str(n["index"])) if source and source != "flat" else None
        pts = [(float(t), float(v) + (0.0 if absolute else tune)) for t, v in (pts or []) if 0 <= t <= n["duration"]]
        if len(pts) >= 2:
            self.mcp.call("set_pitch_curve", note_id=n["id"], points=[{"time_seconds": t, "midi": v} for t, v in pts])
            stable = [v for t, v in pts if t >= 0.3 * n["duration"]] or [v for _, v in pts]
            n["target"] = float(np.median(stable))
            return True
        if abs(tune) > 1e-6:  # 拉平也要对齐伴奏定音
            self.mcp.call("set_pitch_curve", note_id=n["id"], points=[{"time_seconds": 0.0, "midi": n["midi"] + tune},
                                                                     {"time_seconds": n["duration"], "midi": n["midi"] + tune}])
        n["target"] = n["midi"] + tune
        return False

    def _curves(self, source):
        if not source or source == "flat":
            return None
        return json.loads(Path(self.task["pitch_curves"][source]).read_text(encoding="utf-8"))

    def apply_picks(self, picks: dict):
        """（实验用，不暴露给模型）按运行记录里的最终选择直接放素材，用于“同一套素材只改一个变量”的对照。"""
        req, idx = [], []
        for k, pk in picks.items():
            i = int(k)
            req.append({"note_id": self.notes[i]["id"], **{f: pk[f] for f in ("audio_path", "audio_id", "region_id", "alias")}})
            idx.append(i)
        ok, text = self.mcp.call("replace_note_material", roots=self.roots, choices=req,
                                 game_model_dir=str(MODELS / "game" / "medium"), fcpe_model=str(MODELS / "fcpe" / "fcpe.onnx"), **FIT)
        if not ok:
            raise RuntimeError(text)
        for i, k in zip(idx, picks):
            self.mcp.call("set_note", note_id=self.notes[i]["id"], **FLAT)
            self.notes[i]["pick"] = {f: picks[k][f] for f in ("audio_path", "audio_id", "region_id", "alias")}
            self.notes[i]["material"] = picks[k].get("material", "")
        return len(idx)

    def apply_pitch_curves(self, source: str = "reference"):
        """按任务登记的音高线来源画线；source="flat" 表示全部拉平。"""
        if self.legato_done:
            return {"error": "finalize_legato 之后不能再改音高线"}
        srcs = self.task.get("pitch_curves", {})
        if source != "flat" and source not in srcs:
            return {"error": f"任务没有这个音高线来源：{source}；可用 {list(srcs) + ['flat']}"}
        curves = self._curves(source)
        drawn = 0
        for n in self.notes:
            if n.get("material"):
                self.mcp.call("set_note", note_id=n["id"], drift=0, modulation=0)
                drawn += self._draw(n, source, curves)
        self.pitch_source = source
        return {"ok": True, "source": source, "notes_with_curve": drawn, "notes_total": len(self.notes),
                "tuning_offset_cents": round(self._tune() * 100), "hint": "没画线的音保持拉平（参考里那段不可信或没有素材）；音准检查会和画上的线比较，不是和乐谱比"}

    # ---------- Melodyne 式音高（中心 + 部分拉平 + 少量切分），不画逐点曲线 ----------
    def _target_line(self, n: dict, curves):
        tune = self._tune()
        absolute = self.pitch_source in set(self.task.get("pitch_curves_absolute", ["reference"]))
        pts = [(float(t), float(v) + (0.0 if absolute else tune)) for t, v in ((curves or {}).get(str(n["index"])) or [])
               if 0 <= float(t) <= n["duration"]]
        return pts if len(pts) >= 2 else [(0.0, n["midi"] + tune), (n["duration"], n["midi"] + tune)]

    def _note_snapshot(self, note_id: str):
        for t in self.mcp.call_json("project_snapshot")["tracks"]:
            for c in t["clips"]:
                for x in c["notes"]:
                    if x["id"] == note_id:
                        return x
        return None

    # 切分按音高线复杂度：偏差 > tol 才考虑，最优分段（每多一段 penalty 音分·帧），段 ≥ min_piece，台阶 ≥ min_step；不限刀数
    SPLIT = {"tol": 30.0, "min_piece": 0.03, "min_step": 35.0, "penalty": 600.0, "smooth": 0.06}

    def _fit_one(self, base: dict, note_id: str, curves, allow_split: bool, max_splits: int | None = None, snap: dict | None = None):
        x = snap or self._note_snapshot(note_id)
        if x is None:
            return None
        R = self._target_line(base, curves)
        plan = fit_note(x["contour"], min(base["duration"], x["duration_seconds"]), R, max_splits=max_splits if allow_split else 0, **self.SPLIT)
        self.mcp.call("edit_notes_pitch", note_id=note_id, action="set", midi=plan["center_midi"])
        self.mcp.call("set_note", note_id=note_id, drift=plan["drift"], modulation=plan["modulation"])
        made = 0
        cur = note_id; offset = 0.0
        for t_cut, center in plan["splits"]:
            ok, text = self.mcp.call("split_note", note_id=cur, local_seconds=round(t_cut - offset, 4))
            if not ok:
                break
            r = json.loads(text); cur = r["new_note_id"]; offset = t_cut; made += 1
            self.mcp.call("edit_notes_pitch", note_id=cur, action="set", midi=center)
        base["target"] = plan["center_midi"]
        base["target_pieces"] = [(0.0, plan["center_midi"])] + [(tc, c) for tc, c in plan["splits"][:made]]
        return dict(plan, splits_made=made)

    def apply_melodyne_pitch(self, source: str = "reference", allow_split: bool = False, max_splits: int | None = None):
        """所有有素材的音（含两轨交替后的 B 轨）按 Melodyne 式拟合：中心 + 漂移/调制，可选切分。"""
        srcs = self.task.get("pitch_curves", {})
        if source != "flat" and source not in srcs:
            return {"error": f"任务没有这个音高线来源：{source}"}
        self.pitch_mode = "melodyne"; self.pitch_source = source
        curves = self._curves(source)
        snap = self.mcp.call_json("project_snapshot")
        starts = [n["start"] for n in self.notes]
        stats = {"notes": 0, "splits": 0, "flattened_drift": 0, "err": []}
        for t in snap["tracks"]:
            if t.get("muted") or t.get("pitch_algorithm") is None:
                continue
            for c in t["clips"]:
                for x in c["notes"]:
                    k = int(np.argmin([abs(c["start_seconds"] + x["start_seconds"] - s0) for s0 in starts]))
                    base = self.notes[k]
                    if not base.get("pick") or abs(c["start_seconds"] + x["start_seconds"] - base["start"]) > 0.01:
                        continue
                    plan = self._fit_one(base, x["id"], curves, allow_split, max_splits, snap=x)
                    if plan:
                        stats["notes"] += 1; stats["splits"] += plan["splits_made"]
                        stats["flattened_drift"] += plan["drift"] < 0.5; stats["err"].append(plan["err_cents"])
        e = stats.pop("err")
        return {"ok": True, **stats, "mean_err_cents": round(float(np.mean(e)), 1) if e else None}

    def balance_loudness(self, max_db: float = 6.0):
        """按最近一次渲染，把每个音元音段响度拉向全曲中位数（±max_db）。"""
        if self.legato_done:
            return {"error": "finalize_legato 之后不能再调音量"}
        if not self.last_check:
            return {"error": "先 render_and_check 一次，再根据渲染结果平衡音量"}
        y, sr = librosa.load(self.last_check["render"], sr=22050, mono=True)
        song_end = max(n["start"] + n["duration"] for n in self.notes)
        pre = len(y) / sr - song_end
        pre = pre if 0 < pre < 0.6 else 0.0
        loud = {}
        for n in self.notes:
            a = n["start"] + pre + min(0.05, n["duration"] * 0.3)
            b = n["start"] + pre + n["duration"] - 0.02
            seg = y[int(a * sr):int(b * sr)]
            if len(seg) > 200 and n.get("material"):
                loud[n["index"]] = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        if not loud:
            return {"error": "测不到音量"}
        med = float(np.median(list(loud.values())))
        changed = []
        for n in self.notes:
            if n["index"] not in loud:
                continue
            g = float(np.clip(med - loud[n["index"]], -max_db, max_db))
            if abs(g) >= 1.0:
                new_gain = n.get("gain", 1.0) * 10 ** (g / 20)
                self.mcp.call("set_note", note_id=n["id"], gain=round(new_gain, 4))
                n["gain"] = new_gain
                changed.append({"note": n["index"], "db": round(g, 1)})
        spread = float(np.std(list(loud.values())))
        return {"ok": True, "median_db": round(med, 1), "spread_before_db": round(spread, 1), "adjusted": len(changed),
                "largest": sorted(changed, key=lambda c: -abs(c["db"]))[:8], "hint": "再 render_and_check 看效果"}

    def finalize_legato(self, tail_ratio: float = 0.35, tail_min_ms: float = 20.0, edge_fade_ms: float | None = None):
        """收尾：两轨交替 + 元音尾部按音符时长延长 tail_ratio（默认 12.5%，夹在 20～200 ms）并淡出叠到下一个音头上。"""
        if self.legato_done:
            return {"error": "已经做过了"}
        tail_ratio = float(np.clip(tail_ratio, 0.05, 0.5))
        self.final_picks = {n["index"]: dict(n["pick"], material=n["material"]) for n in self.notes if n.get("pick")}
        a_tr, b_tr = split_alternate(self.mcp, self.track_id, dict(RENDER, volume=self.volume))
        info = apply_legato(self.mcp, a_tr, b_tr, tail_ratio=tail_ratio, tail_min_ms=float(tail_min_ms), rebind=self._rebind_extended, edge_fade_ms=edge_fade_ms)
        self.legato_done = True
        return {"ok": True, **info, "tail_ratio": tail_ratio}

    def _rebind_extended(self, resized):
        """延长过的音按新时长重新绑定同一素材（多读真实元音），再恢复拉平/音高线/音量。resized = [(全局序号, 音符快照, 新时长)]"""
        choices, redo = [], []
        for k, snap, new_dur in resized:  # legato 的全局序号与 self.notes 同序（都不含休止符）
            if k >= len(self.notes) or not self.notes[k].get("pick"):
                continue
            base = self.notes[k]
            choices.append({"note_id": snap["id"], **{f: base["pick"][f] for f in ("audio_path", "audio_id", "region_id", "alias")}})
            redo.append((base, snap["id"]))
        if not choices:
            return 0
        ok, text = self.mcp.call("replace_note_material", roots=self.roots, choices=choices,
                                 game_model_dir=str(MODELS / "game" / "medium"), fcpe_model=str(MODELS / "fcpe" / "fcpe.onnx"), **FIT)
        if not ok:
            raise RuntimeError("rebind after tail extension: " + text[:300])
        curves = self._curves(self.pitch_source)
        for base, nid in redo:
            if self.pitch_mode == "melodyne":
                self._fit_one(base, nid, curves, allow_split=False)
            else:
                self.mcp.call("set_note", note_id=nid, **FLAT)
                self._draw(dict(base, id=nid), self.pitch_source, curves)
            if base.get("gain"):
                self.mcp.call("set_note", note_id=nid, gain=round(base["gain"], 4))
        return len(choices)

    def disconnect_all(self):
        """断开音频：每个音符成为独立的块（自己的音频时钟和音高线），挪动或拉长一个音不再压缩/拉伸相邻的音。"""
        done = 0
        for t in self.mcp.call_json("project_snapshot")["tracks"]:
            if t.get("muted") or not t.get("compose", True):
                continue
            ids = [x["id"] for c in t["clips"] for x in c["notes"] if (x.get("label") or "").upper() != "R"]
            if len(ids) >= 2:
                ok, text = self.mcp.call("disconnect_notes", note_ids=ids)
                if not ok:
                    return {"error": text[:200]}
                done += len(ids)
        return {"ok": True, "notes_disconnected": done}

    def render_and_check(self):
        self.renders += 1
        wav = self.run_dir / f"render_{self.renders}.wav"
        ok, text = self.mcp.call("export_wav", path=str(wav), timeout_seconds=600)
        if not ok:
            return {"error": "渲染失败: " + text[:300]}
        gain_note = None
        raw, _ = sf.read(wav, always_2d=True)
        pk = float(np.abs(raw).max())
        if pk > 0.966:  # 超过 -0.3 dBFS：降低轨道音量重新导出，避免削波爆音
            self.volume *= 0.89 / pk
            self.mcp.call("set_track", track_id=self.track_id, volume=self.volume)
            ok, text = self.mcp.call("export_wav", path=str(wav), timeout_seconds=600)
            if not ok:
                return {"error": "渲染失败: " + text[:300]}
            gain_note = f"峰值 {20 * np.log10(pk):+.1f} dBFS，已自动把轨道音量降到 {self.volume:.2f} 后重新导出"
        y, sr = librosa.load(wav, sr=22050, mono=True)
        f0, vf, vp = librosa.pyin(y, fmin=65, fmax=1500, sr=sr, hop_length=220)
        t = np.arange(len(f0)) * 220 / sr
        rms = librosa.feature.rms(y=y, frame_length=882, hop_length=220)[0]
        peak = float(rms.max()) or 1e-9
        # 导出会在 beat 0 之前补上第一个字的辅音（先行），所以整体有一段前置偏移
        song_end = max(n["start"] + n["duration"] for n in self.notes)
        pre = len(y) / sr - song_end
        pre = pre if 0 < pre < 0.6 else 0.0
        per = []
        for n in self.notes:
            a = n["start"] + pre + min(0.08, n["duration"] * 0.3)
            b = n["start"] + pre + n["duration"] - 0.03
            m = (t >= a) & (t < b)
            seg = f0[m & vf]
            loud = float(np.median(rms[: len(m)][m[: len(rms)]])) / peak if m.any() else 0.0
            cents = None
            if len(seg) >= 3:
                pieces = n.get("target_pieces")
                if pieces:  # 切开的音：每个渲染帧和它所在那一段的中心比，不和第一段比
                    local = t[m & vf] - (n["start"] + pre)
                    starts = np.array([q[0] for q in pieces]); centers = np.array([q[1] for q in pieces])
                    expect = centers[np.clip(np.searchsorted(starts, local, side="right") - 1, 0, len(centers) - 1)]
                    cents = round(100 * float(np.median(librosa.hz_to_midi(seg) - expect)))
                else:
                    cents = round(100 * (float(np.median(librosa.hz_to_midi(seg))) - n.get("target", n["midi"] + self._tune())))
            voicing = float(np.nanmean(vp[m])) if m.any() else 0.0
            per.append({"note": n["index"], "lyric": n["char"], "has_material": bool(n["material"]),
                        "pitch_error_cents": cents, "loudness": round(loud, 2), "voicing": round(voicing, 2)})
        if self.lang == "ja":
            heard = None  # 中文 ASR 听不了日语，跳过可懂度
        else:
            y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
            heard = han(asr_model().generate(input=y16)[0].get("text", ""))
        want = [n for n in self.notes if n["char"]]
        want_py = [n["pinyin"] for n in want]
        heard_py = lazy_pinyin(heard, style=Style.NORMAL) if heard else []
        if heard is None:
            want = []
        sm = difflib.SequenceMatcher(a=want_py, b=heard_py, autojunk=False)
        ok_idx = set()
        for a, _, size in sm.get_matching_blocks():
            ok_idx.update(range(a, a + size))
        for k, n in enumerate(want):
            per[n["index"]]["heard_by_asr"] = k in ok_idx
        cov = sum(1 for n in self.notes if n["material"]) / len(self.notes)
        intel = len(ok_idx) / max(1, len(want)) if heard is not None else None
        pitched = [p for p in per if p["pitch_error_cents"] is not None]
        in_tune = sum(1 for p in pitched if abs(p["pitch_error_cents"]) <= 30) / max(1, len(self.notes))
        quiet = [p["note"] for p in per if p["has_material"] and p["loudness"] < 0.08]
        q = quality_metrics(y, sr, self.qref)
        weak_th = 0.6 * self.qref["voicing_mean"] if self.qref else 0.5
        weak = [p["note"] for p in per if p["has_material"] and p["voicing"] < weak_th]
        score_v1 = None if intel is None else round(100 * (0.5 * intel + 0.3 * in_tune + 0.2 * cov))
        if intel is None:
            score = round(100 * (0.3 * in_tune + 0.15 * cov + 0.3 * q["harmonic_norm"] + 0.25 * q["voicing_norm"])
                          - 5 * q["clipped_permille"])
        else:
            score = round(100 * (0.25 * intel + 0.2 * in_tune + 0.1 * cov + 0.25 * q["harmonic_norm"] + 0.2 * q["voicing_norm"])
                          - 5 * q["clipped_permille"])
        problems = [p for p in per if (not p["has_material"]) or (p.get("heard_by_asr") is False)
                    or (p["pitch_error_cents"] is not None and abs(p["pitch_error_cents"]) > 30)
                    or p["note"] in quiet or p["note"] in weak]
        self.last_check = {"render": str(wav), "score": score, "score_v1": score_v1, "asr_heard": heard,
                           "intended": "".join(n["char"] or "＿" for n in self.notes),
                           "intelligibility": None if intel is None else round(intel, 2), "in_tune_ratio": round(in_tune, 2), "coverage": round(cov, 2),
                           "harmonic_ratio": q["harmonic_ratio"], "voicing_mean": q["voicing_mean"],
                           "clipped_permille": q["clipped_permille"],
                           "quality_reference": self.qref}
        if self.best is None or score > self.best["score"]:
            self.best = {"score": score, "render": self.renders,
                         "picks": {n["index"]: n.get("pick") for n in self.notes if n.get("pick")}}
        return {**self.last_check, "best_score_so_far": self.best["score"], "best_render": self.best["render"],
                "gain": gain_note, "quiet_notes": quiet, "weak_notes": weak, "problem_notes": problems,
                "note": ("score 由可懂度(ASR)、音准、覆盖率、谐波占比(越低越像杂音/痰音)、有声概率(越低越虚)组成；"
                         "voicing<0.5 的音标为虚/含糊(weak_notes)。ASR 听拉平的唱歌本来就不准，只作参考。")}

    def finish(self, summary: str):
        restored = None
        if not self.legato_done and self.best and self.last_check and self.last_check["score"] < self.best["score"] and self.best["picks"]:
            last_score = self.last_check["score"]
            picks = self.best["picks"]
            req = [{"note_id": self.notes[i]["id"], **{k: picks[i][k] for k in ("audio_path", "audio_id", "region_id", "alias")}}
                   for i in sorted(picks)]
            self.mcp.call("replace_note_material", roots=self.roots, choices=req,
                          game_model_dir=str(MODELS / "game" / "medium"), fcpe_model=str(MODELS / "fcpe" / "fcpe.onnx"), **FIT)
            for i in sorted(picks):
                self.mcp.call("set_note", note_id=self.notes[i]["id"], **FLAT)
                self.notes[i]["pick"] = picks[i]
            best_render = self.best["render"]
            self.render_and_check()
            restored = f"最后一版 {last_score} 分低于最佳；已自动恢复到第 {best_render} 次渲染的方案并重新渲染（现在 {self.last_check['score']} 分）"
            summary += "\n[系统] " + restored
        self.finished = True
        self.final_summary = summary
        ok, _ = self.mcp.call("project_save", path=str(self.run_dir / "result.hjpx"))
        return {"ok": True, "saved": ok, "restored_best": restored}

    def close(self):
        self.mcp.close()


TOOL_SPECS = [
    {"name": "get_task", "description": "读取任务：填词、速度、每个音符的时间/音高/当前歌词与素材。开始时先调用。",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "assign_lyrics", "description": "把填词的字分配到音符上，每个音符一个汉字。可多次调用覆盖。",
     "parameters": {"type": "object", "properties": {"assignments": {"type": "array", "items": {"type": "object", "properties": {
         "note": {"type": "integer"}, "char": {"type": "string"}}, "required": ["note", "char"]}}}, "required": ["assignments"]}},
    {"name": "search_materials_batch", "description": "一次为 first_note..last_note 每个音符各返回前 top 名候选（精简字段：版本、置信度、辅音长度与占音符比例、元音长度、拉伸倍数、移调半音、综合分）。音符多时用它代替逐个 search_materials。",
     "parameters": {"type": "object", "properties": {"first_note": {"type": "integer"}, "last_note": {"type": "integer"}, "top": {"type": "integer"}},
                    "required": ["first_note", "last_note"]}},
    {"name": "search_phrase", "description": "查找同一条录音里连续说出的一串字（first_note..last_note 的歌词）。连续取用最自然，优先尝试。",
     "parameters": {"type": "object", "properties": {"first_note": {"type": "integer"}, "last_note": {"type": "integer"}},
                    "required": ["first_note", "last_note"]}},
    {"name": "search_materials", "description": "为单个音符找素材候选。返回已按综合分排序的候选及其特征（同字/同音、置信度、元音长度、需要拉伸倍数、移调半音数、原录音说了什么）。",
     "parameters": {"type": "object", "properties": {"note": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["note"]}},
    {"name": "apply_materials", "description": "把选中的候选放到音符上（自动对齐辅音/元音交界、拉平音高）。choices 用 search_materials 的 candidate_id，chain_ids 用 search_phrase 的 chain_id，可同时给。",
     "parameters": {"type": "object", "properties": {
         "choices": {"type": "array", "items": {"type": "object", "properties": {"note": {"type": "integer"}, "candidate_id": {"type": "string"}},
                                                "required": ["note", "candidate_id"]}},
         "chain_ids": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "apply_pitch_curves", "description": "给已放好素材的音符画音高线。source 取任务登记的来源（get_task 的 pitch_curve_sources，如 reference=原唱分离人声、ds_*=DiffSinger 音高模型）或 flat（全部拉平）。",
     "parameters": {"type": "object", "properties": {"source": {"type": "string"}}, "required": ["source"]}},
    {"name": "balance_loudness", "description": "根据最近一次渲染，把每个音的响度拉向全曲中位数（默认最多 ±6 dB）。在素材和音高线都定下来之后做一次，然后再渲染检查。",
     "parameters": {"type": "object", "properties": {"max_db": {"type": "number"}}}},
    {"name": "finalize_legato", "description": "收尾衔接：两轨交替，前一个音的元音尾部按自身时长延长 tail_ratio（默认 0.35，即 35%，范围 0.05～0.5；最多 200 ms，且不超过下一个音长度的一半）并淡出叠到下一个音头上。做完后不能再换素材/音高/音量，只能 render_and_check 和 finish。",
     "parameters": {"type": "object", "properties": {"tail_ratio": {"type": "number"}}}},
    {"name": "render_and_check", "description": "渲染当前工程并自动检查：每个音的音高误差、响度、有声概率(虚)、ASR 能否听出歌词、整段谐波占比(杂音/痰音)、削波，给出 0-100 分、历史最高分和问题音符列表。",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "finish", "description": "完成任务并保存工程。summary 写清楚做了什么、还有哪些已知问题。",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]
