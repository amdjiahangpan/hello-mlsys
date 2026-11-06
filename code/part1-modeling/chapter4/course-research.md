# CH4 课程调研：精度、量化方法与误差预算

> 写作前调研（WRITING_GUIDE 第 2 步）。调研材料本地留存：
> `/tmp/ch4_research/`（CMU 11-868 讲义 PDF+txt、syllabus）、`/tmp/ch4research/`（CS336 lecture .py、UIUC CS498 讲义 PDF+txt、GT CS8803 课表）。
> 调研 agent 两次因网关 500（PDF 图像超限）中断，材料已落盘，本文由主会话直接整理。

## 1. 各课程对量化的覆盖

| 课程 | 量化覆盖 | 深度 | 形式 |
|---|---|---|---|
| **Stanford CS336**（主对标，见 Ch3 调研） | L10「效率」一节：QAT vs PTQ、朴素 absmax 缩放、activation-aware 量化；作为多项效率手段之一（剪枝/蒸馏/推测解码并列） | ~5 分钟口述，无实验、无误差预算 | lecture |
| **CMU 11-868 LLM System 2026 Spring**（Lei Li） | 周 11–12 两讲：L19「Basic methods」（低精度格式、absmax/zero-point、AdaQuant / ZeroQuant / LLM.int8()）；L20「GPTQ」（Hessian 逆 Cholesky、块量化 + 误差补偿）。Reading：GPTQ (ICLR 2023) | 两整讲，方法/算法推导为主，无实测 | slides |
| **UIUC CS498 ML System 2025 Spring**（Minjia Zhang） | 「LLM Quantization」一讲：以 serving 成本（显存/计算）为框架，讲 ZeroQuant 与 SmoothQuant | 一讲，方法为主，无实测 | slides |
| **Georgia Tech CS8803 SAL 2026**（LLM Systems, Inc.） | 课表含多篇量化相关 session（"Quantization for Efficient and Accurate LLM Serving"、"Quantization for LLM Compression and Acceleration" 等，标题见课表快照） | 未取正文（仅课表快照） | 讲者报告式 |
| **Hugging Face llms-course** | 量化章节：bitsandbytes 工具路径（`load_in_8bit` 等），面向使用者 | 工具向，无原理推导、无误差分析 | 教程 |

共同点：全部课程把量化作为**方法主题**讲——格式表、缩放方案（per-tensor/per-channel/absmax/zero-point）、代表算法（GPTQ/SmoothQuant/ZeroQuant/LLM.int8/k-quant），并以 NV/MI 数据中心卡为隐含硬件。共同缺口（即本章的实测层）：

1. **三口径显存对账**（公式账本 × 引擎账本 × 设备读数，残差逐项归因）——课程只给 `N_params × bpw` 一行。
2. **dequant 开销的显式项与 kernel 级校准**——课程把 INT4 的「加速」当卖点，从不拆 GEMV 里的 dequant ALU 成本。
3. **误差预算**：PPL 退化上限（本章预算 1.10）与「哪个约束先被打破」（VRAM/延迟/质量）的判定流程——课程没有。
4. **消费级 RDNA 口径**：无 FP8 预量化样本（9070 XT 有 ai_f8 路径但无 checkpoint 生态）、无 FP4 硬件路径、GGUF k-quant（Q8_0/Q4_K_M）的引擎侧真实布局 vs 公式 bpw 的差——课程默认 H100/A100 + GGUF 只在「下载权重」语境出现。

## 2. 方法与本章小节映射（正文引用锚点）

| 本章内容 | 课程出处（正文引用用） |
|---|---|
| 低精度格式表（fp16/bf16/int8/int4/fp8/fp4 位宽与 range） | CMU L19 格式页；Nagel et al. 2021 white paper（CMU L19 所引，可作二次引用） |
| RTN/absmax/zero-point 朴素量化 | CMU L19、CS336 L10 |
| GPTQ（Hessian 块补偿） | CMU L20 + GPTQ 原论文（ICLR 2023） |
| SmoothQuant / ZeroQuant（激活 outlier 迁移） | UIUC CS498、CMU L19（ZeroQuant 条目） |
| LLM.int8()（mixed 8/16-bit） | CMU L19 |
| k-quant 家族（Q4_K/Q5_K/Q8_0，llama.cpp 实际路径） | 无课程专讲 → 本章以 llama-quantize 报告 + GGUF 文件字节为引擎口径锚点（差异化内容） |
| FP8/FP4 not_covered 说明 | 本机硬件事实（RDNA 路径与生态），非课程观点 |

## 3. 本章定位

沿用课程叙事顺序（格式 → 缩放/方法 → 开销 → 质量），但每一节落到**本机可复现实测**：
- 格式/方法节：引用 CMU/UIUC 定义，落到 GGUF 三档（F16/Q8_0/Q4_K_M）的实际 bpw 与文件体积；
- 开销节：roofline + 显式 dequant 项（Feishu 任务 #2），kernel trace 校准 η_alu；
- 质量节：PPL 预算 1.10（本预算为课程中不存在的「误差预算」概念，正文需自述口径）；
- 断线节：VRAM/TTFT/TPOT/PPL 四预算谁先破（Feishu「哪个约束先被打破」）。
