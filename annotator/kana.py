"""平假名/片假名 → 罗马音（按素材库命名习惯：shi chi tsu fu ji wo n，拗音 kya/sha/cha/ja）。"""
from __future__ import annotations

_BASE = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wi", "ゑ": "we", "を": "wo", "ん": "n", "ゔ": "vu",
}
_YOON = {"ゃ": "a", "ゅ": "u", "ょ": "o", "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}
_YOON_HEAD = {"き": "ky", "ぎ": "gy", "し": "sh", "じ": "j", "ち": "ch", "に": "ny", "ひ": "hy", "び": "by",
              "ぴ": "py", "み": "my", "り": "ry", "ぢ": "j", "ふ": "f", "て": "t", "で": "d", "う": "w", "ゔ": "v"}
# 素材库里常见的变体写法（同音的另一种拼法），检索时一并尝试
ALIASES = {"tsu": ["tu"], "fu": ["hu"], "ji": ["zyi", "dyi"], "shi": ["syi"], "zu": ["du"], "wo": ["o"],
           "chi": ["tyi"], "di": ["dyi"], "ti": ["tyi"]}


def kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def to_romaji(kana: str) -> str | None:
    """单个音拍（可含拗音小字）转罗马音；无法转换返回 None。"""
    s = kata_to_hira(kana.strip())
    if not s:
        return None
    if len(s) == 2 and s[1] in _YOON and s[0] in _YOON_HEAD:
        return _YOON_HEAD[s[0]] + _YOON[s[1]]
    if len(s) == 1 and s in _BASE:
        return _BASE[s]
    out = []
    for ch in s:  # 多字歌词（如 "かい"）逐字拼接
        if ch in _BASE:
            out.append(_BASE[ch])
        else:
            return None
    return "".join(out)


def candidates(romaji: str) -> list[str]:
    return [romaji] + ALIASES.get(romaji, [])
