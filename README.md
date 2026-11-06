# otomad-agent：用 DeepSeek 做音MAD 人力调音

> 这是 HachiShifter fork 里的独立分支 `otomad-agent`，只含 Agent 代码与评测数据。
> HachiShifter 本体的 MCP 新工具在分支 `otomad-mcp-tools`。路径配置见 `otomad_config.py`。

用真人语音采样（主播、角色台词）拼出“唱歌”，流程照搬人工在 Melodyne 里的做法：
参考旋律 → 按填词选素材 → 辅音/元音交界卡在节奏线上、元音拉伸 → 拉到旋律音高 → 听着不对再换素材。

分工：

| 角色 | 做什么 |
|---|---|
| **DeepSeek**（`deepseek-flash`） | 执行：分配歌词、选素材、看检查报告决定怎么修 |
| **Claude** | 训练师：分析失败记录、改 Skill、设计测试与评分 |
| **代码** | 素材自动标注、候选打分、对齐/拉平/渲染（HachiShifter MCP）、自动检查、对照实验 |
| **人** | 盲听打分，校准自动评分 |

## 目录

```
otomad_config.py     路径配置（素材库、exe、模型，可用环境变量覆盖）
annotator/           素材库自动标注：硬链接镜像 → ASR(文件名只作参考) → HubertFA 声韵母对齐 → otto.ini
agent/
  runner.py          DeepSeek 执行器（思考模式 + 工具调用，完整记录 trace）
  tools.py           暴露给 DeepSeek 的 7 个工具 + 自动检查/评分
  skill/SKILL.md     当前 Skill（system prompt）；SKILL_v*.md 为历史版本
  tasks/*.json       测试任务（旋律 + 填词 + 素材文件夹）
  compare.py         对照实验：多个 Skill × 多个任务 × 重复次数
bench/               P 点（辅音/元音交界）对齐器评测：规则法 / HubertFA / SOFA / MFA
pilot/               早期试点脚本与标注/盲听网页生成器
tools/               MCP 客户端、端到端测试、渲染对比
```

不入库（见 `.gitignore`）：素材镜像 `library/`、第三方对齐器与模型 `aligners/`、渲染结果、个人标注与听感数据。

## 依赖

- HachiShifter 开发版（本仓库，分支 `otomad-mcp-tools`）编译产物，需要新增的 MCP 工具：
  `material_query`、`replace_note_material`、`split_note`、`merge_notes`
- 完整运行包里的模型（GAME/FCPE/NSF），路径见 `OTOMAD_MODELS`
- conda 环境 `otomad`（Python 3.11）：`torch`(CPU)、`funasr`、`modelscope`、`pypinyin`、`librosa`、`soundfile`、`openai`、`onnxruntime`、`click`、`textgrid`、`lightning`、`einops`
- HubertFA：`git clone https://github.com/wolfgitpr/HubertFA aligners/HubertFA`，模型 v0.0.7 解压到 `aligners/hfa_model/`
- 环境变量 `DEEPSEEK_API_KEY`

## 关键结论（实验记录）

- **P 点**：HubertFA + 14 ms 偏移，在 29 个盲标样本上与人工中位差 8 ms、±30 ms 内 90%；
  同一人两次标注中位差 23 ms——已达人耳一致性。零声母不标辅音。SOFA（唱歌模型）在语音上明显更差。
- **文件名只能参考**：很多文件录了比文件名更多的话，整句标注以 ASR 为准，文件名作热词。
- **渲染**：盲听结论 world + 音高完全拉平最好；world 下 stretch_algorithm 不生效。
- **评分**：只看 ASR + 音准会在同一任务的版本间排错；v2 加入谐波占比（杂音/痰音）、有声概率（虚）、削波（爆音）。

## 运行

```bash
# 标注一个素材文件夹（写到 library/ 镜像，不改原素材）
D:/miniconda/envs/otomad/python.exe annotator/annotate.py "棍の音/棍音源"

# 跑一个调音任务
cd agent && D:/miniconda/envs/otomad/python.exe runner.py tasks/t02_duanwei.json

# Skill 对照实验
cd agent && D:/miniconda/envs/otomad/python.exe compare.py --skills skill/SKILL_v1.md skill/SKILL.md --repeat 2
```
