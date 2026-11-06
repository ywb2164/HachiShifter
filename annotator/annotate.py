"""素材库自动标注：镜像 → 文本（文件名拼音 / ASR）→ HubertFA 声韵母对齐 → otto.ini。

    D:/miniconda/envs/otomad/python.exe annotate.py <素材库子目录> [--mode auto|single|sentence] [--limit N]

- 原始素材库只读。镜像建在 otomad-agent/library/<相对路径>：wav 用硬链接（不占空间、不改音频），
  mp3 解码成同名 wav。otto.ini 只写在镜像里，通过 HachiShifter MCP 的 sample_settings_save 写入。
- 单字：文件名就是拼音（如 kan.wav、a2.wav）时直接用；否则按整句处理。
- 整句：seaco-paraformer 听写实际内容，文件名里的汉字只作热词；按拼音和文件名对齐，
  两边一致的字用文件名写法（保留梗写法），只在录音里的字照样保留但降低置信度。
- P：HubertFA 声母音素结束 + 14 ms（盲标校准值）；零声母 P = 元音起点，辅音长度 0。
- 置信度：HubertFA 与规则法（低频能量）相差 ≤30 ms 记 0.9，否则 0.6；仅 ASR 听到的字再 ×0.7。
- 另存 library/<子目录>/annotations.json，给调音 Agent 检索用（时长、P、音高中位数、置信度、来源文本）。
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import otomad_config as cfg  # noqa: E402

import argparse
import difflib
import json
import os
import re
import sys
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pypinyin import Style, lazy_pinyin

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
LIB = cfg.LIBRARY
MIRROR = AGENT / "library"
HFA_DIR = AGENT / "aligners" / "HubertFA"
HFA_MODEL = AGENT / "aligners" / "hfa_model" / "1218_hfa_model_new_dict"
EXE = str(cfg.EXE)
P_OFFSET = 0.014
INITIAL_PHONES = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r",
                  "z", "c", "s", "y", "w"}
NON_SPEECH = {"SP", "AP", "EP", ""}
SYLLABIC = {"n": "en", "ng": "en", "m": "en", "hm": "heng", "hng": "heng"}  # 嗯/呣/哼 这类无元音拼音

sys.path.insert(0, str(HFA_DIR))
sys.path.insert(0, str(AGENT / "tools"))


def fl(x: float) -> float:
    return int(max(0.0, x) * 10000) / 10000


def han(s: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", s))


def load_hfa():
    cwd = os.getcwd()
    os.chdir(HFA_DIR)
    try:
        from onnx_infer import InferenceOnnx
        from tools.g2p import DictionaryG2P

        inf = InferenceOnnx(onnx_path=HFA_MODEL / "model.onnx")
        inf.load_config()
        inf.init_decoder()
        inf.load_model()
        lang = "zh" if inf.vocab["language_prefix"] else None
        g2p = DictionaryG2P(lang, HFA_MODEL / "ds-zh-pinyin-lite.txt")
    finally:
        os.chdir(cwd)
    valid = {line.split("\t")[0] for line in (HFA_MODEL / "ds-zh-pinyin-lite.txt").read_text(encoding="utf-8").splitlines() if "\t" in line}
    return inf, g2p, valid


def mirror(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".wav":
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
        return dst
    out = dst.with_suffix(".wav")
    if (src.with_suffix(".wav")).exists():  # 同名 wav 已存在，避免覆盖
        out = dst.with_name(dst.name + ".wav")
    if not out.exists():
        y, sr = librosa.load(src, sr=None, mono=False)
        sf.write(out, y.T if y.ndim > 1 else y, sr, subtype="PCM_16")
    return out


def reconcile(asr: str, name: str):
    a_py = [lazy_pinyin(c, style=Style.NORMAL)[0] for c in asr]
    n_py = [lazy_pinyin(c, style=Style.NORMAL)[0] for c in name]
    sm = difflib.SequenceMatcher(a=a_py, b=n_py, autojunk=False)
    out, dropped = [], ""
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out += [(name[j1 + k], n_py[j1 + k], "both") for k in range(i2 - i1)]
        else:
            out += [(asr[i], a_py[i], "asr_only") for i in range(i1, i2)]
            dropped += name[j1:j2]
    return out, dropped


def energy_bounds(y: np.ndarray, sr: int) -> tuple[float, float]:
    hop = 110
    rms = librosa.feature.rms(y=y, frame_length=441, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    idx = np.where((db > db.max() - 40) & (db > -60))[0]
    if len(idx) == 0:
        return 0.0, len(y) / sr
    return idx[0] * hop / sr, min(len(y) / sr, (idx[-1] + 1) * hop / sr)


def low_band_rise(y: np.ndarray, sr: int, a: float, b: float) -> float | None:
    hop = 55
    S = np.abs(librosa.stft(y, n_fft=256, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=256)
    low = 10 * np.log10(np.maximum(S[(f >= 150) & (f < 1000)].sum(axis=0), 1e-12))
    low = np.convolve(low, np.ones(3) / 3, mode="same")
    i0, i1 = int(a * sr / hop), min(len(low) - 1, int(b * sr / hop))
    if i1 <= i0:
        return None
    hit = np.where(low[i0:i1 + 1] >= low[i0:i1 + 1].max() - 6)[0]
    return (i0 + int(hit[0])) * hop / sr if len(hit) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subdir")
    ap.add_argument("--mode", default="auto", choices=["auto", "single", "sentence"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=90.0)
    args = ap.parse_args()

    src_root = LIB / args.subdir
    dst_root = MIRROR / args.subdir
    files = sorted(p for p in src_root.rglob("*") if p.suffix.lower() in (".wav", ".mp3") and "__MACOSX" not in p.parts)
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} 个音频：{src_root}")

    inf, g2p, valid = load_hfa()
    asr = fa = None
    jobs, skipped = [], []
    for src in files:
        rel = src.relative_to(src_root)
        wav = mirror(src, dst_root / rel)
        try:
            dur = sf.info(wav).duration
        except Exception as e:
            skipped.append((str(rel), f"读不了: {e}"))
            continue
        if dur > args.max_seconds:
            skipped.append((str(rel), f"太长 {dur:.0f}s（合集类，先跳过）"))
            continue
        syl = re.sub(r"[^a-z]", "", src.stem.lower().replace("ü", "v"))
        is_single = args.mode == "single" or (args.mode == "auto" and syl in valid and not han(src.stem))
        if is_single:
            chars = [(syl, syl, "filename")]
            asr_text, dropped = "", ""
        else:
            if asr is None:
                from funasr import AutoModel
                asr = AutoModel(model="paraformer-zh", disable_update=True)
            y16, _ = librosa.load(wav, sr=16000, mono=True)
            hot = " ".join(dict.fromkeys(re.findall(r"[\u4e00-\u9fff]{2,}", re.sub(r"[（(\[【].*?[)）\]】]", "", src.stem))))
            res = asr.generate(input=y16, hotword=hot)[0] if hot else asr.generate(input=y16)[0]
            asr_text = han(res.get("text", ""))
            if not asr_text:
                skipped.append((str(rel), "ASR 没听出字（叫声/音效/非中文）"))
                continue
            chars, dropped = reconcile(asr_text, han(src.stem))
        chars = [(ch, SYLLABIC.get(py, py), kind) for ch, py, kind in chars]
        pys = [py for _, py, _ in chars]
        bad = [py for py in pys if py not in valid]
        if bad:
            skipped.append((str(rel), f"拼音不在对齐词典里: {bad}"))
            continue
        ph_seq, word_seq, idx = g2p(" ".join(pys))
        fa_starts = None
        if not is_single:
            if fa is None:
                from funasr import AutoModel
                fa = AutoModel(model="fa-zh", disable_update=True)
            text = "".join(ch for ch, _, _ in chars)
            try:
                st = fa.generate(input=(y16, " ".join(text)), data_type=("sound", "text"))[0].get("timestamp") or []
                fa_starts = [a / 1000 for a, _ in st] if len(st) == len(chars) else None
            except Exception:
                fa_starts = None
        jobs.append({"wav": wav, "rel": str(rel), "chars": chars, "asr": asr_text, "dropped": dropped,
                     "name": src.name, "hfa_in": (wav, ph_seq, word_seq, idx), "fa_starts": fa_starts})

    print(f"对齐 {len(jobs)} 个，跳过 {len(skipped)} 个")
    inf.dataset = [j["hfa_in"] for j in jobs]
    inf.predictions = []
    cwd = os.getcwd()
    os.chdir(HFA_DIR)
    try:
        inf.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)
    finally:
        os.chdir(cwd)
    pred = {str(p[0]): p[2] for p in inf.predictions}

    from mcp_client import HachiMcp
    mcp = HachiMcp(EXE, [f"--roots={MIRROR}"])
    index = []
    written = 0
    for j in jobs:
        words = [w for w in pred.get(str(j["wav"]), []) if w.text not in NON_SPEECH]
        if len(words) != len(j["chars"]):
            skipped.append((j["rel"], f"对齐结果字数不符 {len(words)}/{len(j['chars'])}"))
            continue
        y, sr = librosa.load(j["wav"], sr=22050, mono=True)
        audio_id = "a" + uuid.uuid4().hex
        ids = ["p" + uuid.uuid4().hex for _ in words]
        rows = []
        for k, (w, (ch, py, src_kind)) in enumerate(zip(words, j["chars"])):
            phs = [p for p in w.phonemes if p.text not in NON_SPEECH]
            start, end = phs[0].start, phs[-1].end
            zero = phs[0].text not in INITIAL_PHONES or len(phs) == 1
            p = start if zero else phs[0].end + P_OFFSET
            notes = []
            if len(words) == 1:
                # 单字：区域覆盖整个发音（能量包络），HubertFA 在超短片段上常把字尾判早
                e_start, e_end = energy_bounds(y, sr)
                hfa_len = end - start
                start, end = min(start, e_start), max(end, e_end)
                if hfa_len < 0.6 * (end - start):
                    notes.append("HubertFA 字长明显短于实际发音")
            else:
                nxt = words[k + 1].phonemes[0].start if k + 1 < len(words) else len(y) / sr
                if end - p < 0.04:
                    end = max(end, min(nxt, p + 0.08))
            total = int(len(y) / sr * 10000) / 10000 - 0.0005  # 不能超出录音（取整误差也不行）
            start, end = max(0.0, start), min(end, total)
            if zero:
                p = start  # 零声母不需要辅音段（像あいうえお）
            linked = False
            if len(words) > 1 and k + 1 < len(words):
                nxt_start = [p2 for p2 in words[k + 1].phonemes if p2.text not in NON_SPEECH][0].start
                if abs(nxt_start - end) < 0.03:
                    end, linked = nxt_start, True  # 首尾严格相接，HachiShifter 才认作连续素材
            p = min(max(start, p), max(start, end - 0.03))
            if len(words) == 1:
                rule = low_band_rise(y, sr, max(0.0, start - 0.03), end)
                if notes and rule is not None and not zero:
                    p = min(max(start, rule), max(start, end - 0.03))
                    notes.append("P 改用规则法")
                agree = rule is None or zero or abs(rule - p) <= 0.030
            else:
                fs = j["fa_starts"]
                agree = fs is not None and abs(fs[k] - start) <= 0.06
                if not agree:
                    notes.append("两个对齐模型对字头判断不一致" if fs is not None else "没有第二个模型的结果可核对")
            conf = 0.9 if agree and not any(n.startswith("HubertFA") for n in notes) else 0.6
            if src_kind == "asr_only":
                conf *= 0.7
            seg = y[int(p * sr):int(end * sr)]
            f0med = None
            if len(seg) > 2048:
                f0, vf, _ = librosa.pyin(seg, fmin=65, fmax=1500, sr=sr, frame_length=1024, hop_length=220)
                if vf.any():
                    f0med = round(float(librosa.hz_to_midi(np.nanmedian(f0[vf]))), 2)
            row = {"name": py, "region_start_seconds": fl(start), "region_end_seconds": fl(end),
                   "alignment_seconds": fl(p), "fixed_duration_seconds": fl(p - start),
                   "material_audio_id": audio_id, "material_region_id": ids[k],
                   "native_provenance": "otomad-hfa", "native_confidence": round(conf, 2)}
            if ch != py:
                row["additional_aliases"] = [ch]
            if linked:
                row["connected_next_region_id"] = ids[k + 1]
            rows.append(row)
            index.append({"file": j["rel"], "name_in_library": j["name"], "pinyin": py, "char": ch if ch != py else "",
                          "source": src_kind, "start": round(start, 4), "end": round(end, 4), "p": round(p, 4),
                          "consonant_ms": round((p - start) * 1000, 1), "vowel_ms": round((end - p) * 1000, 1),
                          "median_midi": f0med, "confidence": round(conf, 2), "region_id": ids[k], "notes": notes,
                          "asr_text": j["asr"], "filename_not_heard": j["dropped"]})
        ok, text = mcp.call("sample_settings_save", audio_path=str(j["wav"]), rows=rows)
        if ok:
            written += 1
        else:
            skipped.append((j["rel"], "写 otto 失败: " + text[:200]))
    mcp.close()
    (dst_root / "annotations.json").write_text(json.dumps({"rows": index, "skipped": skipped}, ensure_ascii=False, indent=1), encoding="utf-8")
    conf = [r["confidence"] for r in index]
    print(f"写入 {written} 个文件、{len(index)} 个发音；置信度≥0.9 的 {sum(c >= 0.9 for c in conf)} 个；跳过 {len(skipped)} 个")
    for s in skipped[:15]:
        print("  跳过", s)


if __name__ == "__main__":
    main()
