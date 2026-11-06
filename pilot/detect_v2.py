"""P 点检测 v2：同时算多种候选，供人工标注后校准。

沿用 results.json 里同一批 80 个样本（audio/<id>.wav 已是单声道副本）。
每个样本输出：
  - 无边距频谱图 img2/<id>.png（横轴像素与时间线性对应，网页上可以直接点选时间）
  - 候选 P（秒）：
      pyin_onset   旧方法：pyin 连续有声起点（窗口 46 ms，偏早）
      low_rise10   低频段(150-1000Hz)能量首次达到元音峰值 -10 dB
      low_rise6    同上 -6 dB
      low_rise3    同上 -3 dB
      low_slope    pyin 起点附近低频能量上升最快处
      hf_drop      高频段(3-8kHz)辅音噪声相对低频首次让位处
  - 当前展示给你的 P = low_rise6（等你的标注来定哪种最好、要不要加偏移）
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
SR = 22050
HOP = 55  # 2.5 ms
NFFT = 256  # 11.6 ms 窗


def band_db(S, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    e = S[m].sum(axis=0)
    return 10 * np.log10(np.maximum(e, 1e-12))


def candidates(y: np.ndarray, start: float, end: float) -> dict:
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP, center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    low = band_db(S, freqs, 150, 1000)
    hf = band_db(S, freqs, 3000, 8000)
    low = np.convolve(low, np.ones(3) / 3, mode="same")
    t = np.arange(len(low)) * HOP / SR
    i0, i1 = int(start * SR / HOP), min(len(low) - 1, int(end * SR / HOP))

    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1500, sr=SR, frame_length=1024, hop_length=110)
    vt = np.arange(len(voiced)) * 110 / SR
    run, onset = 0, None
    for k, v in enumerate(voiced):
        if vt[k] < start:
            continue
        run = run + 1 if v else 0
        if run >= 4:
            onset = vt[k - 3]
            break
    out = {"pyin_onset": None if onset is None else round(float(onset), 4)}
    if i1 <= i0:
        return out
    peak = low[i0:i1 + 1].max()
    for name, drop in (("low_rise10", 10), ("low_rise6", 6), ("low_rise3", 3)):
        hit = np.where(low[i0:i1 + 1] >= peak - drop)[0]
        out[name] = round(float(t[i0 + hit[0]]), 4) if len(hit) else None
    if onset is not None:
        a, b = max(i0, int((onset - 0.04) * SR / HOP)), min(i1, int((onset + 0.12) * SR / HOP))
        if b > a + 2:
            d = np.diff(low[a:b + 1])
            out["low_slope"] = round(float(t[a + int(np.argmax(d)) + 1]), 4)
    ratio = hf - low
    hit = np.where(ratio[i0:i1 + 1] < -6)[0]
    out["hf_drop"] = round(float(t[i0 + hit[0]]), 4) if len(hit) else None
    return out


def plot(y: np.ndarray, dest: Path) -> None:
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=HOP)), ref=np.max)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    keep = freqs <= 8000
    fig = plt.figure(figsize=(10, 2.6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(np.arange(S.shape[1]) * HOP / SR, freqs[keep], S[keep], shading="auto", cmap="magma", vmin=-80, vmax=0)
    ax.set_xlim(0, len(y) / SR)
    ax.set_ylim(0, 8000)
    ax.axis("off")
    fig.savefig(dest)
    plt.close(fig)


def main() -> None:
    rows = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    (OUT / "img2").mkdir(exist_ok=True)
    for r in rows:
        y, _ = librosa.load(OUT / "audio" / f"{r['id']}.wav", sr=SR, mono=True)
        r["duration"] = round(len(y) / SR, 4)
        c = candidates(y, r["start"], r["end"])
        r["cands"] = c
        r["p_v1"] = r.get("p")
        r["p"] = c.get("low_rise6") or c.get("pyin_onset")
        plot(y, OUT / "img2" / f"{r['id']}.png")
        print(r["id"], r["file"], {k: v for k, v in c.items()})
    (OUT / "results_v2.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    d = [(r["p"] - r["p_v1"]) * 1000 for r in rows if r.get("p") is not None and r.get("p_v1") is not None]
    print(f"\n新 P 相对旧 P 平均后移 {np.mean(d):.1f} ms（中位 {np.median(d):.1f} ms）")


if __name__ == "__main__":
    main()
