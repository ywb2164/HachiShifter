"""整句切字 v2：录音内容以 ASR 为准，文件名只作参考。

    D:/miniconda/envs/otomad/python.exe segment_v2.py [数量] [随机种子]

和 v1 的区别：
  1. 先听写：seaco-paraformer 识别实际内容，文件名里的汉字作为热词（只提高权重，不强制）
  2. 按拼音对齐 ASR 与文件名：
       两边都有 → 用文件名的写法（保留“尼玛/草”这类梗写法），来源 = both（高置信）
       只在录音里 → 保留 ASR 的字，来源 = asr_only
       只在文件名里 → 丢弃，记入 dropped（录音里没找到）
  3. 用对齐后的文本做 fa-zh 字级时间戳
  4. 字内 P：低频(150-1000Hz)能量首次达到该字峰值 -6 dB（单字标注里 ±15 ms 命中率最高的方法）
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
import matplotlib
import numpy as np
import soundfile as sf
from pypinyin import Style, lazy_pinyin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_sentences import pick, split_initial  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "Microsoft YaHei"
OUT = Path(__file__).resolve().parent
LIB = cfg.LIBRARY
SR, HOP, NFFT = 22050, 55, 256


def han(s: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", s))


def reconcile(asr: str, name: str):
    """返回 [(字, 拼音, 来源)] 和 文件名中没在录音里找到的字。"""
    a_py = lazy_pinyin(asr, style=Style.NORMAL, errors="ignore")
    n_py = lazy_pinyin(name, style=Style.NORMAL, errors="ignore")
    if len(a_py) != len(asr):
        a_py = [lazy_pinyin(c, style=Style.NORMAL)[0] for c in asr]
    if len(n_py) != len(name):
        n_py = [lazy_pinyin(c, style=Style.NORMAL)[0] for c in name]
    sm = difflib.SequenceMatcher(a=a_py, b=n_py, autojunk=False)
    chars, dropped = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            chars += [(name[j1 + k], n_py[j1 + k], "both") for k in range(i2 - i1)]
        else:
            chars += [(asr[i], a_py[i], "asr_only") for i in range(i1, i2)]
            dropped += list(name[j1:j2])
    return chars, "".join(dropped)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    from funasr import AutoModel

    fa = AutoModel(model="fa-zh", disable_update=True)
    asr = AutoModel(model="paraformer-zh", disable_update=True)
    (OUT / "sent_audio").mkdir(exist_ok=True)
    (OUT / "sent_img").mkdir(exist_ok=True)
    out = []
    for k, (f, name_text) in enumerate(pick(n, seed)):
        sid = f"t{k:03d}"
        wav16 = OUT / "sent_audio" / f"{sid}.wav"
        if not wav16.exists():
            y16, _ = librosa.load(f, sr=16000, mono=True)
            sf.write(wav16, y16, 16000, subtype="PCM_16")
        hot = " ".join(dict.fromkeys(re.findall(r"[\u4e00-\u9fff]{2,}", re.sub(r"[（(\[【].*?[)）\]】]", "", f.stem))))
        res = asr.generate(input=str(wav16), hotword=hot)[0] if hot else asr.generate(input=str(wav16))[0]
        asr_text = han(res.get("text", ""))
        base = {"id": sid, "file": f.name, "folder": str(f.parent.relative_to(LIB)), "name_text": name_text,
                "text": asr_text, "asr_text": asr_text}
        if not asr_text:
            out.append({**base, "status": "ASR 没听出任何字（可能是叫声/音效）", "chars": [], "dropped": name_text})
            print(sid, f.name, "ASR 空")
            continue
        recon, dropped = reconcile(asr_text, name_text)
        text = "".join(c for c, _, _ in recon)
        fa_res = fa.generate(input=(str(wav16), " ".join(text)), data_type=("sound", "text"))[0]
        stamps = fa_res.get("timestamp") or []
        if len(stamps) != len(recon):
            out.append({**base, "text": text, "status": "对齐失败", "chars": [], "dropped": dropped})
            continue

        y, _ = librosa.load(f, sr=SR, mono=True)
        S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) ** 2
        freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
        low = 10 * np.log10(np.maximum(S[(freqs >= 150) & (freqs < 1000)].sum(axis=0), 1e-12))
        low = np.convolve(low, np.ones(3) / 3, mode="same")
        fr = lambda t: int(round(t * SR / HOP))  # noqa: E731
        tm = lambda i: round(i * HOP / SR, 4)  # noqa: E731

        chars, prev_end = [], 0.0
        for (ch, py, src), (s_ms, e_ms) in zip(recon, stamps):
            s, e = s_ms / 1000, e_ms / 1000
            start = max(prev_end, s - 0.03)
            ini, kind = split_initial(py)
            i0, i1 = fr(start), min(len(low) - 1, fr(e))
            reasons, p = [], None
            if i1 > i0:
                peak = low[i0:i1 + 1].max()
                hit = np.where(low[i0:i1 + 1] >= peak - 6)[0]
                p = tm(i0 + int(hit[0])) if len(hit) else None
            if p is None:
                reasons.append("找不到元音起点")
            elif (e - p) < 0.04:
                reasons.append("元音段短于 40 ms")
            if e - s < 0.06:
                reasons.append("字长短于 60 ms")
            if src == "asr_only":
                reasons.append("文件名里没有这个字（仅 ASR 听到）")
            if ini in {"d", "g", "h", "n", "l", "m"}:
                reasons.append(f"{ini} 声母：单字标注中 P 偏早较多")
            chars.append({"char": ch, "pinyin": py, "source": src, "initial": ini, "initial_kind": kind,
                          "start": round(start, 4), "end": round(e, 4), "p": p,
                          "consonant_ms": None if p is None else round((p - start) * 1000, 1),
                          "verdict": "存疑" if reasons else "自信", "reasons": reasons})
            prev_end = e

        fig, ax = plt.subplots(figsize=(max(6.0, min(16.0, len(y) / SR * 2.2)), 2.8), dpi=80)
        Sd = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=110)), ref=np.max)
        fq = librosa.fft_frequencies(sr=SR, n_fft=1024)
        keep = fq <= 8000
        ax.pcolormesh(np.arange(Sd.shape[1]) * 110 / SR, fq[keep], Sd[keep], shading="auto", cmap="magma", vmin=-80, vmax=0)
        for c in chars:
            ax.axvline(c["start"], color="white", lw=1, ls="--")
            ax.axvline(c["end"], color="white", lw=0.6, ls=":")
            if c["p"] is not None:
                ax.axvline(c["p"], color="#ffd400", lw=1.5)
                ax.axvspan(c["start"], c["p"], color="#ffd400", alpha=0.15)
            col = "#7dd3a0" if c["source"] == "both" else "#ff8a65"
            ax.text((c["start"] + c["end"]) / 2, 7300, c["char"], color=col, ha="center", fontsize=11)
        ax.set_ylim(0, 8000)
        ax.set_title(f"{sid}  文件名「{name_text}」  ASR「{asr_text}」", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "sent_img" / f"{sid}.png")
        plt.close(fig)
        play = OUT / "sent_audio" / f"{sid}_play.wav"
        if not play.exists():
            mono, sr0 = librosa.load(f, sr=None, mono=True)
            sf.write(play, mono, sr0, subtype="PCM_16")

        both = sum(c["source"] == "both" for c in chars)
        out.append({**base, "text": text, "status": "ok", "chars": chars, "dropped": dropped,
                    "asr_match": round(both / max(1, len(name_text)), 2)})
        print(sid, f.name, f"录音 {len(chars)} 字（与文件名一致 {both}，仅录音 {len(chars) - both}），文件名中未找到「{dropped}」")
    (OUT / "sentences.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    allc = [c for s in out for c in s["chars"]]
    print(f"\n句子 {len(out)}，字 {len(allc)}，两边一致 {sum(c['source'] == 'both' for c in allc)}，"
          f"仅录音 {sum(c['source'] == 'asr_only' for c in allc)}，自信 {sum(c['verdict'] == '自信' for c in allc)}")


if __name__ == "__main__":
    main()
