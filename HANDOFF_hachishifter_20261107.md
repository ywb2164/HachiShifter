# HachiShifter（HCS）现状交接清单 · 2026-11-07

给接手优化卡顿的模型看。所有数字都是本机实测；没有标注"已验证"的结论请先复核再用。

## 1. 代码与构建

| 项 | 位置 / 做法 |
|---|---|
| 源码（dev 207 + 本地改动） | `D:\HachiShifter\HachiShifter-dev-latest-20261105`，git 分支 `otomad-mcp-tools`（remote `fork` = github.com/ywb2164/HachiShifter；**`origin` 是上游 openhachimi，不要 push**） |
| 构建 | `D:/HachiShifter/otomad-agent/tools/cmake/cmake/data/bin/cmake.exe --build build-integrated --config RelWithDebInfo --target HachiShifterNext --parallel`（Git Bash 下别用 `/m`，会被当路径） |
| 产物 | `build-integrated\HachiShifterNext_artefacts\RelWithDebInfo\HachiShifter Next.exe`（带 pdb，可做栈采样） |
| 用户日常开的 dev 版 | `D:\HachiShifter\HachiShifter-dev-run\启动-dev.cmd`（独立设置目录 `settings\`）。exe 已换成当前编译版，旧版备份 `HachiShifter Next.exe.bak-20261105` |
| 源文件编码 | CRLF，编辑时保持 CRLF |
| Agent 代码 | `D:\HachiShifter\otomad-agent`（独立 git，分支 `otomad-agent`）；素材库 `D:\杠氏老方` 只读，镜像在 `otomad-agent\library` |

**全部改动都还没提交。** `git diff --stat` 涉及：`AudioEngine.cpp`、`Mld5ContinuousSource.h`、`Mld5Renderer.cpp/.h`、`NativeAudioFocus.h`、`TimelineComponent.cpp`、`backend/McpServer.cpp`、`Main.cpp`（一个旧的拖动 smoke，与本次无关）。

## 2. 已做的性能修复（已验证，MLD5 输出逐字节不变）

| 问题 | 位置 | 修复 | 效果（雨爱.mpd，745 音符，2 个 MP3 源） |
|---|---|---|---|
| MLD5 缓存键把同源整段 session 原样写进每个片段的 key（O(N²)，key 是原始字节串） | `AudioEngine.cpp` `writeMld5SourceSessionContext` / `renderKey` | 每次 rebuild 按 (轨, 源文件) 只算一次 SHA-256 摘要（`Mld5SessionDigestScope`） | 无头导出 53 s / 4.5 GB 且失败 → 16 s / 0.49 GB 成功 |
| 超过 ±48 半音的音让整轨 MLD5 报错无声 | `Mld5Renderer.cpp` `preparePitchCarrier` | 比例钳在 1/16..16 | 整轨可渲染 |
| `writeNativePitchContext` 对每个片段把整条轨所有片段 `expandedClipParts` 复制一遍 | `AudioEngine.cpp` | 先判断 pitch group，不可能匹配的不展开 | 消息线程 rebuild 大幅缩短 |
| 时间轴每帧深拷贝整条轨 + O(N²) 焦点透明度 | `TimelineComponent.cpp`、`NativeAudioFocus.h` | 只在拖动时拷贝；批量算透明度；只画可见区域；重叠带排序扫描 | 播放时 UI 线程占用下降 |
| MLD5 连续源规划：每个片段对所有旧 bucket 复制+重规划 | `AudioEngine.cpp`（bucket 循环）、`Mld5ContinuousSource.h` `ownerAt` | 精确预检（目标间隙/身份/源偏移/包含）；`ownerAt` 改游标；push/pop 代替复制 | 编辑后无响应 15 s+（MCP 超时）→ 约 6 s |

## 3. 仍然卡的地方（建议优先级从高到低）

1. **`AudioEngine::rebuildLoadedClips` 在消息线程同步执行**（`MainComponent::syncAudio` → `syncProject`）。大工程每次编辑都卡 UI 数秒；MCP live 请求因此超时（"Editor disconnected or request timed out"）。最根本的解法是把 rebuild 移出消息线程（快照 → 后台构建 → 原子替换 loadedClips）。
2. **`planMld5ContinuousSource` 对一条真正连续的长串，每加一个片段就整串重新校验+合并（O(k²)）**。现在占 rebuild 的大头（采样里约 70%）。可做增量校验：只校验新片段及其与前一片段的成对条件，最后每个 bucket 只合并一次（合并失败再回退原算法）。
3. **钢琴窗每帧做文字排版**：`PianoRollComponent::paint` → `drawFittedText` → HarfBuzz（COLR 表 sanitize）。播放时约占一个核心。可缓存 GlyphArrangement 或只在标签变化时排版。
4. **`project_snapshot` 太大**：745 音符约 20 MB JSON、live 下 7–10 s（音高轮廓 + source_time_map 占绝大部分）。给 agent 用需要精简查询（按时间段 / 按音符、可选不带轮廓）。
5. **WORLD 每次渲染都重新做完整分析**（Harvest + D4C + CheapTrick），没有按 (源, 区段) 缓存；Melodyne 工程本身已带 F0，可直接用。
6. **NSF-HiFiGAN**：每个 ORT session `intraOpThreads = CPU−1`，渲染队列再并行 → 雨爱 1100–1290% CPU、约 1 GB，播放前要等约 60 s。导入 Melodyne 工程默认用 NSF（用户设置已改为 MLD5，原设置备份 `%APPDATA%\HachiShifterNext\HachiShifterNext.settings.bak-before-mld5-import-20261107`）。
7. **每个 FileRenderJob 都新开一个音频 reader**；MP3 走 Windows Media Foundation（`WMCreateSyncReader`），745 个任务各自打开同一个 MP3。
8. **恢复文件**：雨爱的自动恢复文件 18 MB，周期性整份重写。
9. `transport_play` 在 live MCP 里本身要 3–7 s 才返回（被 rebuild 阻塞）。
10. **Bug**：`import_audio` 路径含中文时返回 `ok; notes=1` 但不建轨（ASCII 路径正常）。

复现素材：`D:\30949\Documents\雨爱.mpd`（素材齐全）。**不要反复拿它做已知的占用测试**；做之前先确认要验证的是新问题。小工程对照：`C:\Users\30949\Downloads\teio\teio\teio.mpd`（81 音符，HCS 渲染与 Melodyne 导出逐样本一致）。

## 4. MCP 新增 / 改动（fork 分支，未提交）

`material_query`、`replace_note_material`（新增 `vowel_fit: stretch|trim`、`max_vowel_compression`，返回 `vowel_trimmed`）、`split_note`、`merge_notes`、`disconnect_notes`（= 编辑器「断开音频」）、`move_native_notes`（= 编辑器拖动，含相邻拉伸）、`import_melodyne` / `project_open` 新增 `pitch_algorithm`、`set_track` 新增 `native_audio_overlap`（快照里也输出）。

## 5. MLD5 宿主适配（音质，不是卡顿，供参考）

- 同一个 teio 工程：Melodyne 导入路径的 HCS 渲染 == Melodyne 输出（已验证）。
- 已修：素材绑定把整段元音压进音符（最多 14 倍）→ `vowel_fit=trim`；`resize_note` 不会多读素材音频（agent 侧改为延长后重新绑定）；绑定后的音符是一整块（挪一个音会拉伸相邻音）→ 用 `disconnect_notes`；允许音频重叠时伸出的尾巴保持自己的音高（`applyNativeSharedPitch` 的 `holdOwnPitch`）。
- **仍未解决的"电音"**：把 teio 按 agent 的素材绑定路径重建（同样素材、区段、音高、增益），与 Melodyne 路径对比：交给 MLD5 的有声帧比例 **0.86 vs 0.97**（无声帧直接走未移调的原声，音符内部在处理/原声间来回切换），时间映射 2 点线性 vs 约 117 点；输出有声概率 0.482 vs 0.515、谐波占比 0.789 vs 0.810。下一步应查 GAME+FCPE 源音高分析的有声判定（`AnalysisService` / `completeMaterialSourceAnalysis`）为何比 Melodyne 保守，以及 `Mld5Renderer.cpp` 里的有声门控（`gate`）在频繁切换时的处理。重建脚本：`otomad-agent/agent/teio_rebuild.py`。

## 6. 调试工具

| 工具 | 用法 |
|---|---|
| `HACHI_MLD5_DUMP=<目录>` | 每个 MLD5 请求写出输入 wav、控制数组（源/目标音高、时间映射、增益）、输出 wav（`Mld5Renderer::render`） |
| 栈采样 | 草稿目录 `stacksample.py <pid> <秒> <间隔ms>`（dbghelp + pdb，Python ctypes） |
| GUI 探针 | `gui_probe.py`：起 GUI → live MCP 打开工程、改音、播放，记录 CPU/内存；用 `HACHI_TEST_SETTINGS_DIR` 隔离设置，避免用户的恢复文件弹窗 |
| 盲听页 | `otomad-agent/pilot/label_server.py`（8765 端口）+ `listen_day1107.html`；结果在 `pilot/labels_day1107_pref.json` |

## 7. 注意事项

- 不要用 `taskkill /IM` 之类按名字杀进程：用户可能开着自己的 HachiShifter；只按 PID 杀自己启动的。
- 强杀带未保存工程的 GUI 会在 `%APPDATA%\HachiShifterNext\Recovery` 留恢复文件，下次启动弹窗；测试用 `HACHI_TEST_SETTINGS_DIR`。
- 用户最在意的是效果和听感；性能优化不能改变渲染结果（每步都用导出 wav 的 md5 对比）。
