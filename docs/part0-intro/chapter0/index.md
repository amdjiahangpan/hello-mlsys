---
title: "第 0 章 协议与纸面演练"
description: "Hello MLSys · 第 0 章 协议与纸面演练 · 教程定位、硬件图谱、统一测量协议、模拟器工具链与能力边界"
---

# 第 0 章 协议与纸面演练

::: danger ⚠️ Alpha内测版本警告
此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。
:::

## 本章导读

> 本章不装环境，不跑训练，也不调 kernel。我们先坐下来把四件事说清楚：**这本书为谁解决什么问题**、**你手里有什么硬件和模型**、**测量要遵守什么规则才可信**、**模拟器能做什么、不能做什么**。读完并完成本章实验后，你手里应该有一份字段完整的纸面实验底稿——它是后续每一章真机实验的模板，也是全书第一条"预测 → 实测 → 校准"闭环的起点。

## 教学目标

- 能说明本教程与 hello-rocm（跑起来）、hello-gpu（算子做快）、MLSys·im（系统建模）的分工，并复述"预测 → 实测 → 校准 → 决策"的学习闭环；
- 能列出目标 AMD 卡的算力、显存带宽、显存容量与互联参数，并注明参数来源，区分**标称值**与**实测有效值**；
- 能执行统一测量协议：token / GB-GiB / 延迟 / 功耗口径，预热、重复次数与方差记录，校准集与留出配置的预先划分；
- 能说清模拟器工具链能建模什么（性能、显存、排队、通信）、不能建模什么，以及缺项时最小教学分析模型的定位；
- 能填写一份字段完整的纸面实验底稿（硬件版本、ROCm 版本、预热次数、重复次数、留出配置），并完成一次纯纸面的权重显存与 Roofline 手算。

## 0.1 教程定位与学习路径

### 为什么需要一门 ML Systems

先从一个观察开始。业界常说："世界正匆忙地建造 AI 系统，但并没有在*工程化*地建造它们（The world is rushing to build AI systems. It is not engineering them.）"（mlsysbook.ai, 第 1 章）。这句话的意思是：大量 AI 系统是在试错中搭起来的——"先跑起来再说，慢了再调"。CMU 15-442/15-642 课程开篇用同一个问题切入：**为什么需要 ML Systems？** 答案是：模型效果之外，**系统决定了这个模型能不能上线、成本多少、尾延迟是否达标**——而系统行为可以用物理量（FLOPs、字节数、带宽、延迟）来预测，而不是只能靠试。

这给了本书一个可以操作的第一定律。mlsysbook.ai 称之为 **Iron Law of ML Systems**：

```text
T ≈ D / BW  +  O / (R · η)  +  L
     └─数据项─┘  └───计算项───┘  └─开销项─┘
  搬运数据的下界  计算的下界(打折)  调度/启动/通信
```

- **D / BW**：这一步要搬运多少字节（权重、KV），除以带宽，是时间下界；
- **O / (R·η)**：这一步要做多少 FLOPs，除以峰值算力再除以**利用率 η**——η 是你离硬件极限的距离；
- **L**：kernel 启动、调度、通信等固定开销。

注意：roofline 引擎实际取的是 `max(数据项, 计算项) + 开销项`——一步至少花掉"算完"或"搬完"中较慢的那个时间（whichever is larger），而不是两者相加。你会在第 1 章反复用到这个公式：每一次"预测"都是它在具体模型、硬件、配置下的实例化；每一次"校准"都是在问"我的 η 和 L 猜得对不对"。

### 这本书和它的姐妹教程

&emsp;&emsp;datawhale 的这条线里，每本教程回答一个不同的问题：

| 教程 | 回答的问题 | 本书与它的关系 |
| --- | --- | --- |
| **hello-rocm** | 环境怎么装，程序怎么跑起来 | 环境安装与 ROCm 部署一律引用它，本书不复述 |
| **hello-gpu** | 单个算子为什么慢、怎么优化（kernel 级） | 算子/kernel 调优是它的主场；本书把算子当"黑盒成本"，只关心它的 FLOPs 与字节数 |
| **MLSys·im** | 系统建模的方法与现成模拟器（roofline、372 个公式、102 份硬件参数） | **参考模拟器**：本书解析层借它的方法与公式，兼作对照学生手算的第三方标尺 |
| **mlsys-simulator**（自研） | 课程自研的确定性 CPU 服务模拟器——服务级性能怎么估（continuous batching、KV 内存压力、TP/PP、Prefix Cache、Chunked Prefill、多副本路由） | 本书的主工具（§1.1/§1.4 的 simulate/predict 六元组接口即它）；解析层借 MLSys·im 的方法与公式；v1.15 起支持声明式量化权重/KV 存储与反量化开销；私有仓库，许可证待定 |
| **hello-mlsys（本书）** | **应该设计什么系统、怎么知道它有效** | 用模拟指导系统设计，在 AMD 真机上验证和校准 |

一句话分工：**hello-rocm 教你跑起来，hello-gpu 教你把算子做快，MLSys·im（mlsysim）提供广泛使用的系统建模方法与参考模拟器，hello-mlsys 用自研 msys-simulator（确定性 CPU 服务模拟器）教你用模拟指导系统设计，并在 AMD 真机上验证和校准。** 训练显存优化、K8s 集群调度属于平台侧，作为选读指向 hello-ai-infra-platform，不在本书主线。

### 全书的学习闭环：预测 → 实测 → 校准 → 决策

&emsp;&emsp;每章都围绕同一个七步闭环展开。它和"先跑起来再说"的区别在于：**预测必须发生在测量之前，并留痕**。

```text
定义场景与约束 → 运行模拟器并留存预测 → 选择实测点 → AMD 真机验证
        ↑                                              │
        └────────── 设计决策 ← 留出验证 ← 解释误差与校准 ┘
```

四条纪律（本书反复引用，此处只定义一次）：

1. **预测先于测量，且留痕。** 先看了测量值再"拟合"出的预测只能叫回拟合（back-fitting），不算预测。
2. **校准集拟合，留出集只评测。** 用留出结果回头再调参，留出集就烧掉了，必须换新留出配置——"先看了答案再填参数"无效。
3. **重复 ≠ 泛化。** 同一个配置跑 3 轮测的是**方差**，不能宣称"第 1、2 轮校准、第 3 轮验证"。
4. **结论必须带证据等级。** `mock`（LLM 当模拟器，只走流程）< `simulated`（真 DES 模拟器 + 校准参数）< `measured`（真机）；`hardware_validated` 结论要求留出集上的真机证据。

### 这本书适合谁，前置要求是什么

**前置要求**：

| 要求 | 说明 |
| --- | --- |
| Python / PyTorch | 能读懂推理服务代码（如 vLLM 的启动参数），不必会写 kernel |
| Linux 命令行 | 会装环境、跑脚本、看日志；熟悉 `git` |
| 英文文档阅读 | ROCm / vLLM 上游文档以英文为准 |
| 基础概率统计 | 理解均值、p99、方差，能看懂"50 个样本的 p99 说明不了什么" |

**读者画像**：

| 如果你…… | 这本书给你…… |
| --- | --- |
| 用 PyTorch 能跑模型，但说不清系统性能从哪来 | 一套"把系统行为翻译成 FLOPs/字节/带宽"的方法 |
| 想进 MLSys / AI Infra 方向 | 与 CMU 15-442 大纲对齐的建模主线 + 可复现的校准实验 |
| 做容量规划、量化选型、框架选型，缺判断依据 | 每章一个可量化的决策：显存账本、误差预算、α-β 通信、SLA 容量 |
| 想以 AMD GPU 为主力学习环境 | 9070 XT 为主线的完整真机路径 + W7900D 集群章 + 无设备替代路径 |

**边界声明**：

- 不是为了反 NVIDIA——问题是平台无关的，换 NVIDIA 卡只是换参数，方法论不变；
- 不是 kernel 调优手册——那是 hello-gpu；
- 不承诺"一次预测精确到 ±5%"——本书的目标是**可信的量级 + 被校准的误差 + 有证据的决策**，精度来自校准而不是承诺。

### 学习路线图

| 部分 | 章节 | 你会完成什么 | 对应课程周次 |
| --- | --- | --- | --- |
| **Part 0** | 第 0 章（本章） | 填出一份字段完整的纸面实验底稿（含纯纸面手算） | 课前 + 第 1 周前半 |
| **Part 1** 建模与校准 | 第 1 章 | Roofline 预测：矩阵乘 / LayerNorm / FlashAttention 的瓶颈判定与时间量级 | 第 1 周 |
| | 第 2 章 | 第一次真机校准：拟合 η_compute、η_mem、T_launch，留出集验证 | 第 2 周 |
| | 第 3 章 | 显存账本：权重 + KV 缓存，容量上限与并发上限 | 第 3 周 |
| | 第 4 章 | 量化误差预算：位宽、算子敏感度、质量-性能折中 | 第 3–4 周 |
| | 第 5 章 | 通信建模：α-β 模型，PCIe/链路带宽实测 | 第 4 周 |
| | 第 6 章 | 模型并行：TP/PP 的显存与通信代价 | 第 4–5 周 |
| | 第 7 章 | 服务排队：连续批处理、SLA 容量边界 | 第 5 周 |
| | 第 8 章 | 框架取舍：vLLM / SGLang 等的系统级对比 | 第 5 周 |
| **Part 2** 与 AI 一起设计 | 第 9.1–9.3 章 | 三个实战：端侧部署助手、单机 SLA、W7900D 多卡集群设计 | 第 6–8 周 |
| **Part 3** | 选读与案例库 | KV 搬运 vs 重算等专题 + 案例卡 | 随时 |
| **附录 A–F** | 硬件卡 / 模型卡 / 术语 / 参考 / 社区 / 排错 | 全书查询手册 | 随时 |

## 0.2 AMD GPU 硬件图谱

### 教学档位 T1–T4

&emsp;&emsp;全书实验按四个档位标注适用性；**每个实验页面都会标出"适用档位 + 无设备替代路径"**，没有对应硬件也能完成该章的模拟器部分。

| 档位 | 硬件 | 角色 | 状态 |
| --- | --- | --- | --- |
| **T1** | RX 9070 XT 16 GB（单卡） | 主线真机：Part 1 全部实验 + 9.1 / 9.2 | 参数已核实（下表） |
| **T2** | W7900D 48 GB（单卡） | 适配 / 云端：T1 装不下或需更大显存的场景 | 型号待资源方确认（TODO：@嘉航） |
| **T3** | W7900D 48 GB 多卡集群（≥4 卡） | 第 9.3 章：32B + 7B deep-research 负载的集群设计 | 同上 |
| **T4** | 无 GPU / 云端 | 替代路径：纯 CPU 预测（Part 1 全部可完成）或云端 ROCm 环境 | 始终可用 |

### 为什么选 AMD

&emsp;&emsp;理由和 hello-gpu 一致，这里只说结论：

- **消费级卡买得起**：9070 XT 是学生和小型实验室真能买到的卡，真机校准不依赖机房；
- **生态年轻是特性**：文档缺口、工具链差异会逼你查一手资料——这正是 MLSys 要练的能力；
- **问题平台无关**：Roofline、显存账本、α-β 通信在 NVIDIA 上同样成立，换卡只换参数表。
- **消费卡没有数据中心互联**：9070 XT 多卡只有 PCIe，这反而让第 5 章"通信吃掉多少并行收益"变得非常具体——PCIe 上的 TP 代价高昂，逼你做真实的取舍。

### RX 9070 XT（T1）关键参数

&emsp;&emsp;下表数字全部来自 **AMD 官方产品页**（来源列可点击核对）；"推导"列是本教程为教学做的换算，不是 AMD 给的数。**真机有效值以第 2 章校准为准，标称值只用于第 1 章的预测。**

| 参数 | 标称值 | 来源 / 备注 |
| --- | --- | --- |
| 架构 | RDNA 4（Navi 48），64 CU / 4096 SP | [AMD 官方页](https://www.amd.com/en/products/graphics/desktop/rx-9000/rx-9070-xt.html) |
| 加速频率 | ≤ 2970 MHz（游戏频率 2400 MHz） | AMD 官方页 |
| FP32 向量峰值 | 48.7 TFLOPS | AMD 官方页 |
| FP16 矩阵峰值 | 195 TFLOPS（矩阵指令路径） | AMD 官方页 |
| FP8 / INT8 矩阵 | 389 TFLOPs / 389 TOPs | AMD 官方页 |
| 显存 | 16 GB GDDR6，256-bit，20 Gbps | AMD 官方页 |
| 显存带宽 | 640 GB/s（256-bit × 20 Gbps 标称） | 由上式推导；**实测见第 2 章** |
| 主机互联 | PCIe 5.0 x16 | AMD 官方页 |
| 板卡功耗 | 304 W | AMD 官方页 |
| ROCm / 驱动版本 | 实验时记录（`rocminfo`、`rocm-smi`） | 每次真机实验先做环境快照再运行 |

**教学用 ridge 点**（推导值，第 1 章 Roofline 图的纵轴交点）：

- FP16 矩阵路径：195e12 / 640e9 ≈ **305 FLOP/byte**
- FP32 向量路径：48.7e12 / 640e9 ≈ **76 FLOP/byte**

&emsp;&emsp;AI 高于 ridge 点 → Compute-bound；低于 → Memory-bound。**注意这些 ridge 点用的是标称带宽**；第 2 章拟合出有效带宽后，ridge 点会移动——这正是"预测 → 校准"的第一次显形。

### W7900D（T2 / T3）

| 参数 | 值 | 状态 |
| --- | --- | --- |
| 显存 | 48 GB | 已确认（第 9.3 章场景定义） |
| 集群规模 | ≥ 4 卡 | 已确认（32B + 7B deep-research 负载） |
| 精确型号 / 算力 / 带宽 / 互联 | —— | **待资源方确认后回填**（TODO：@嘉航）；回填前 W7900D 相关实验一律先走 T4 纯 CPU 预测路径 |

### 模型档位

&emsp;&emsp;全书主线使用 Qwen3 系列（附录 B 模型卡记录完整架构参数，来源为各模型 Hugging Face `config.json`）：

| 模型 | 架构要点（HF config） | 用在哪 | 为什么 |
| --- | --- | --- | --- |
| **Qwen3-0.6B** | 28 层，hidden 1024，16 q-heads / 8 kv-heads，head_dim 128，embedding 与 LM head 共享 | 第 1/2 章建模与校准 | 最小、最快，profile 友好，校准集首选 |
| **Qwen3-1.7B** | 28 层，hidden 2048，16 / 8 heads，inter 6144 | 第 2 章 kernel 归因（认知用途）、第 4/5/7 章对比 | 中档，暴露更多算子结构 |
| **Qwen3-4B** | 36 层，hidden 2560，32 / 8 heads，inter 9728 | 第 7/8 章服务负载 | 16 GB 显存下最直观的主力档位 |
| Qwen3-7B / 32B | —— | 第 9.3 章集群 | 32B 单卡装不下 → 强制做出切分决策 |

::: tip 命名口径
模型名统一使用 Hugging Face 仓库 ID：`Qwen/Qwen3-0.6B` / `Qwen/Qwen3-1.7B` / `Qwen/Qwen3-4B`（0.6B 为 0.59 B 参数）。引用实验脚本与模型卡时以仓库 ID 为准。
:::

### 真机验证命令与环境快照

&emsp;&emsp;每次真机实验**先记录环境，再运行**。最小快照（写入底稿 `environment` 字段）：

```bash
rocm-smi --showproductname --showmeminfo vram   # 卡型、显存总量
rocminfo                                        # 驱动 / ROCm 版本、GPU 型号
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
vllm --version                                  # 若使用 vLLM
```

规则：**配置不可用时不静默改配置**——标注 `unsupported`，保留报错与版本号，供第 2 章误差归因引用（软件版本、ROCm 版本都是归因项）。

### 无设备替代路径（T4）

| 场景 | 替代 |
| --- | --- |
| 无 9070 XT | 第 1 章（Roofline 预测）与第 3 章（显存账本）**纯 CPU 可完成**，只缺第 2 章真机校准 |
| 需要 W7900D 而未到位 | 先以 T1 参数跑模拟器，底稿标注 `pending: W7900D spec`，回填后重跑 |
| 完全无 GPU | 云端 ROCm 环境（如 AMD Radeon Cloud）；测量协议不变，`config_id` 中的 device 字段如实填写 |

## 0.3 统一测量协议

### 为什么需要协议

&emsp;&emsp;性能工作里最大的假优化来源不是算法，而是**测量方式**："真的更快了，还是只是换了量法？"三个典型漂移：

- **单位混用**：16 GB 写成 16000 MB 或 15.26 GiB，显存账本对不上；
- **没有预热**：首请求承担冷启动、缓存未命中、JIT 编译成本，TTFT 偏差一个数量级；
- **先看答案再拟合**：在测量集上拟合参数再自称"验证"——那叫回拟合。

协议的目的不是流程繁琐，而是让**任何一章、任何人、任何时候的测量结果都可以互相比较**。完整协议见 `curriculum/measurement-protocol.md`；本节是每章必须遵守的最小集。

### 单位口径

| 项 | 口径 |
| --- | --- |
| token 数 | 按实际 tokenizer + chat template 计算（含 system prompt 与模板），分别记录"原始输入长度"与"完整输入 token 数" |
| 容量（显存等） | 二进制 **GiB**（2³⁰ 字节）——与 `rocm-smi`（MiB ÷ 1024）、PyTorch 分配报告对齐 |
| 带宽 / 互联速率 | 十进制 **GB/s**（10⁹ B/s）——与标称规格口径一致 |
| 首次出现 | 必须同时标注单位；同一实验内禁止混用（如"权重 1.19 GB" 与 "KV 112 KiB/token" 可以并存，因为都标了单位，但同一列必须统一） |
| 延迟 | ms；TTFT 从客户端提交时刻起算（含网络 + 排队）；TPOT = (末段到达 − 首段到达)/(N−1)，仅 N>1 且正常完成时报告，标注为客户端估计 |
| 功耗 | W（板卡级），记录采样间隔；**降频（throttling）是误差归因项**，不是噪音 |
| 可复现性 | 随机种子、软件版本、拓扑必须记录在底稿 |

### 四种身份标识

&emsp;&emsp;测量记录的唯一键由四个 ID 构成（`measurement-protocol.md` §6）：

| ID | 含义 | 不变量 |
| --- | --- | --- |
| `trace_id` | 请求内容 / 顺序 / tokenizer / 模板 / 生成目标的不可变指纹 | 换了任何一个 → 新 trace |
| `config_id` | 模型 / 设备 / 版本 / 引擎参数 / 并发 / 开关的组合指纹 | 底稿 `scenario + environment` 的指纹 |
| `run_id` | 一次完整重放（30 预热 + 150 测量） | 重跑不覆盖，新 run 新 ID |
| `request_id` | trace 内稳定的请求序号 | 预热与测量段分开标注 |

预测记录额外挂 `prediction_id`、模拟器版本、校准版本，与 `config_id` 绑定——**预测和测量靠 `config_id` 对齐，靠时间戳先后关系判定"预测先于测量"**。

### 预热、重复与方差

- **10 条预热 + 50 条测量**；整条 trace 完整重放 **3 轮**（每配置 = 30 预热 + 150 测量）；
- 预热单独记账，不计入测量；每轮开始时恢复到相同初始状态（服务重启或显式声明 KV/缓存状态）；
- 报告样本数、失败数、超时数；
- **50 条请求的 p99 接近最慢样本，不能作为稳定尾延迟或 SLA 容量的结论**——要报 p99 必须另给样本预算；
- 单请求超时 180 s，整轮预算 900 s；失败轮保留，重跑用新 `run_id`，**不得覆盖原始记录**；
- 固定并发池为**闭环**（closed-loop）：实际到达时间取决于完成时间。模拟器必须执行相同的补位（refill）规则，否则"预测 vs 实测"差的是协议而不是系统；想测开环排队需要独立的开环 trace。

### 校准集与留出集

- 测量**之前**冻结：场景、候选矩阵、预测指标、参数来源、校准/留出清单——全部写进底稿；
- **校准集用于拟合，留出集只用于评测**；用留出结果回头再调参，留出集即烧掉，必须换新留出配置；
- 同一配置跑 3 轮测的是**方差**，不得宣称"第 1、2 轮校准、第 3 轮验证"；
- 模拟器超出外推域时：默认停住并标注（`unsupported` / `out_of_domain`），不编造置信区间。

### 四个入口：predict / measure / compare / calibrate

&emsp;&emsp;全书实验共用四类入口（对应 `code/` 下的脚本，具体命名以最终工具发布为准）：

```text
predict   模型 + 配置 → 预测指标（测前留存，带 prediction_id 与时间戳）
measure   真机 / 模拟器后端 → 逐请求 + 系统指标（带 run_id）
compare   预测 vs 实测 → 误差 + 归因 + 结论（证据等级由两侧最高者决定）
calibrate 校准集实测 → 新的参数版本（η_compute / η_mem / T_launch，必须过留出集）
```

&emsp;&emsp;本节固定"测量侧"的口径，下一节固定"预测侧"的工具链；两者共同落进 0.5 的底稿。

## 0.4 模拟器工具链与能力边界

### 预测从哪来：三层结构

&emsp;&emsp;本书的模拟器（开发计划见 `simulator-development-plan.md`）分三层，**公式化而不是写死几个固定数字**：

```text
┌─ 校准层：标称值 → 真机有效值（η_compute / η_mem / T_launch），带版本与适用域
├─ DES 层：请求到达 / 准入 / batch 组织 / KV 占用 / 排队 / 多副本
│          状态机；只感知 prefill 步与 decode 步，时间由解析层给出
└─ 解析层：一步 = 宏观组件分解（GEMM 类 / Attention 类 / Element-wise 类）
           每组件：工作量(FLOPs, 字节) → Roofline → 时间
           t_step = Σ max(T_compute, T_memory) + T_fixed   ← Iron Law 实例
```

- **解析层**回答"一步要多久"：GEMM 类工作量随输入 token 数 O(S)（prefill）或全量权重扫描（小 batch decode，天然 memory-bound）；Attention 类 O(S²)（prefill）或 KV 全读（decode）；Element-wise 类（RMSNorm / RoPE / 激活）按带宽下限计；
- **DES 层**回答"一条请求从提交到完成要多久"：它调用解析层的 `predict_prefill(batch_tokens, …)` / `predict_decode_step(batch_size, context_lengths, …)` 获取步时间，**不下探到 kernel 级**；
- **校准层**回答"标称值和真机差多少"：第 2 章在两个固定阶段（定长 Prefill、定并发 Decode）的端到端时间上拟合三个参数，并用未见过的 workload 配置做留出验证。

&emsp;&emsp;运行入口采用六元组契约，五个产物全部落盘：

```text
simulate(workload, infra, strategy, calibration, evaluation, execution)
→ manifest.json / result.json / requests.jsonl / events.jsonl / logs.txt
```

`workload` 是 trace，`infra` 是硬件档位（T1–T4），`strategy` 是调度/切分策略，`calibration` 是参数版本，`evaluation` 定义指标与 SLA，`execution` 记录种子、模拟器版本与预算。

### 能力边界表

| 机制 | 状态 | 依赖章节 |
| --- | --- | --- |
| 性能：prefill / decode 步时间 | ✅ 解析模型 + 真机校准 | 第 1、2 章 |
| 显存：权重 / KV 账本 | ✅ 解析模型 | 第 3、4、9.1 章 |
| 排队 / batch 组织 / 连续批处理 | ✅ DES（V0.5） | 第 7、9.2 章 |
| 通信：α-β 模型（AllReduce 等） | 🚧 模型已定义；**真机链路参数待校准** | 第 5、6 章 |
| TP / PP 切分 | 🚧 建设中（V1.0 范围） | 第 6、9.3 章 |
| 抢占 / KV 驱逐 | 🚧 建设中 | 第 7（归因）、9.3 |
| 前缀缓存（PagedAttention/Radix 风格） | 🚧 建设中 | 第 7、9.2 |
| 投机解码 / MoE 路由 | ⬜ 选读（案例卡） | Part 3 |
| **答案质量** | ❌ `not_evaluated`——模拟器不判质量，质量必须真评 | 第 4、9.1 章 |
| kernel 级执行 / 框架调度器内部 | ❌ 黑盒，只经测量观察 | 第 8 章 |

&emsp;&emsp;一条硬规则：**不支持的机制显式拒绝运行或返回 `not_covered`，不静默忽略参数**（`E_MECHANISM_UNSUPPORTED`）。宁可报错，不可假装支持——"未核实项不写成上游已支持"。

### 证据等级与结论等级

| 证据等级 | 含义 | 能支撑的结论 |
| --- | --- | --- |
| `mock` | LLM 自己当模拟器（量级粗，只为走通流程） | `mock_only` |
| `simulated` | 真 DES 模拟器 + 校准参数 | `simulation_validated`（需说明参数适用域） |
| `measured` | AMD 真机 | `hardware_validated`（需留出集 + 证据与局限说明） |
| —— | 预测与实测差过大且归因未闭合 | `unresolved`（如实报告，不强行下结论） |

### 模拟器告诉不了你的事（以及怎么办）

- **质量**：量化掉点多少、提示词变了体验如何——必须真评（第 4 章误差预算 + 第 9.1 章评测集）；
- **框架调度器的真实行为**：模拟器建模的是"应有的批处理语义"，真实 vLLM/SGLang 的调度细节以测量为准（第 8 章）；
- **kernel 级性能**：某个 kernel 在 RDNA4 上有没有跑满——hello-gpu 的主场，本书只引用其结论。

&emsp;&emsp;预测与实测**分歧时**，按四类偏差归因：测量噪音 / 参数漂移（η、L 过期）/ 机制缺失（`not_covered` 被漏标）/ 实现语义差异（模拟器与真机的 refill、批处理规则不一致）。第 2 章第一次完整走一遍这个归因流程。

&emsp;&emsp;上游缺项时用**最小教学分析模型**补齐：要求可解释（一页纸能推完）、有手算示例（本章 0.5.3 就是模板）、显式声明适用范围。

## 0.5 纸面实验底稿

### 为什么先填底稿

&emsp;&emsp;闭环第一步"预测先于测量并留痕"，如果没有纸面记录，就不存在闭环。底稿是全书统一的记录载体：**每个实验一份底稿，字段固定，每次重跑新建目录，不覆盖历史**。本章的底稿是纯纸面演练；从第 2 章起，同一模板承载真机实验。

### 底稿模板

完整模板见 [`experiment-draft.yaml`](/part0-intro/chapter0/experiment-draft.yaml)（源码位于 `docs/public/part0-intro/chapter0/`，点击可查看或复制；内含填写说明注释与一个已填好的示例，后续各章底稿共用此模板）。顶层字段：

| 字段 | 填什么 |
| --- | --- |
| `experiment_id` / `chapter` | 底稿编号与所属章节 |
| `hypothesis` | 你认为会发生什么 + 为什么，不是实验步骤 |
| `scenario` | 模型 / 硬件档位 / 负载——`config_id` 的指纹 |
| `environment` | 真机快照（§0.2 命令）或 `paper-only` |
| `prediction` | 预测工具、参数版本、预测指标；`stored_at` 必须先于测量 |
| `measurement_plan` | 预热 / 样本 / 轮次 / 超时；校准集与留出集（测前冻结） |
| `acceptance` | 误差阈值与决策规则 |

后续章节使用时整体替换示例值，字段结构保持不变；重跑实验新建底稿文件，不覆盖历史。

### 手算示例：Qwen3-0.6B 的一次 decode 步 + 一份显存账本

&emsp;&emsp;这就是"纸面演练"的全部内容——只用手头已核实的参数（§0.2 硬件表 + HF config），在模拟器就位之前先把量级算出来。第 1 章的模拟器输出会重跑同一场景，**手算 vs 模拟器的差异本身成为第一条校准记录**。

**架构事实**（Qwen/Qwen3-0.6B，HF `config.json`）：28 层；hidden 1024；16 q-heads / 8 kv-heads；head_dim 128（注意 Q 输出维度 = 16×128 = 2048 ≠ hidden）；intermediate 3072；embedding 与 LM head 共享；BF16。

**第 1 步：参数计数 → 每步权重字节数（decode 的 GEMM 类工作量）**

| 块 | 每层参数 | ×28 层 |
| --- | --- | --- |
| Q (1024→2048) + K,V (1024→1024×2) + O (2048→1024) | 2.10M + 1.05M + 1.05M + 2.10M ≈ 6.29M | 176.1M |
| FFN (gate/up/down，1024×3072×3) | 9.44M | 264.2M |
| LayerNorm（忽略不计） | ~2K | ~0.1M |
| **层小计** | **15.73M** | **440.4M** |
| embedding + LM head（共享，151,936×1024） | —— | 155.6M |
| **总计** | | **≈ 596M 参数** |

decode 每步的 GEMM 权重扫描 = 全部层参数 + LM head 投影（embedding 表本身只按 1 个 token 查表，不算扫描）：

```text
每步权重字节 ≈ (440.4M + 155.6M) × 2 B ≈ 1.19 GB（十进制）≈ 1.11 GiB（二进制）
t_weight = 1.19e9 B ÷ 640 GB/s ≈ 1.85 ms   （标称带宽，η = 1）
```

**第 2 步：Attention 类 = decode 的 KV 全读**

```text
KV / token = 2 (K,V) × 28 层 × 8 kv-heads × 128 dim × 2 B = 114,688 B = 112 KiB/token
ctx = 1024 → KV 读取 = 112 KiB × 1024 = 112 MiB（≈ 117 MB 十进制）
t_kv = 0.117e9 B ÷ 640 GB/s ≈ 0.18 ms
QK^T / Softmax·V 的 FLOPs ≈ 数十 MFLOP，对 195 TFLOPS 可忽略 → Attention 类 memory-bound
```

**第 3 步：瓶颈判定与时间下界**

```text
AI(GEMM 类) ≈ 1.19 GFLOP ÷ 1.19 GB ≈ 1.0 FLOP/byte  ≪  ridge 305 → Memory-bound
t_step ≈ t_weight + t_kv + T_launch ≈ 1.85 + 0.18 + T_launch(ms，未校准)
       ≈ 2.0–2.3 ms  →  decode 上限 ≈ 430–500 tok/s（B=1，标称，无开销折扣）
```

&emsp;&emsp;这条结论（**decode B=1 是教科书级 memory-bound：一步 ≈ 一次全量权重扫描**）就是底稿 `hypothesis` 字段的来源；第 2 章的真机测量会告诉你真实的 η 是多少。

**第 4 步：显存账本（同场景的容量面）**

```text
可用显存 = 16 GiB × 0.80（gpu_memory_utilization）= 12.8 GiB
权重占用 ≈ 1.11 GiB
框架 / 激活开销 ≈ 0.5 GiB（假设值，声明；真机以第 2 章 profile 为准）
KV 预算 ≈ 12.8 − 1.11 − 0.5 ≈ 11.2 GiB
KV 容量 ≈ 11.2 GiB ÷ 112 KiB/token ≈ 105,000 token
→ 例如 50 并发 × 2048 ctx 可容纳；10 并发 × 10,000 ctx 可容纳
```

### 验收清单

- [ ] `hypothesis` 写的是"我认为会发生什么 + 为什么"，而不是实验步骤；
- [ ] `prediction.stored_at` 时间戳早于任何测量开始；
- [ ] 容量用 GiB、带宽用 GB/s，同一列不混用；
- [ ] 四个 ID（trace/config/run/request）与 `prediction_id` 齐全；
- [ ] 校准集与留出配置**已冻结**，留出集未参与任何拟合；
- [ ] 重跑产生新 `run_id`，原始记录未被覆盖；
- [ ] 不支持的机制标注了 `not_covered`，没有静默改配置。

## 实验设计

### 模拟器实验

完成 0.5.3 的纸面手算并填写一份字段完整的底稿（即本章交付物）。第 1 章模拟器就位后，用模拟器输出重跑同一场景（Qwen3-0.6B，T1，decode B=1/ctx=1024），将"手算 vs 模拟器"的差异作为第一条**预测对预测**的校准记录存档。

### 真机实验：无

本章不使用真机；底稿 `environment` 字段填 `paper-only`。真机验证与 Profile 自第 2 章开始。

### 闭环与交付

- **预测（测前留存）**：0.5.3 手算的 decode 步时间与显存账本，连同底稿字段（硬件版本、ROCm 版本占位、预热 10、重复 3 轮 × 50、留出配置）留存。
- **实测（选点运行）**：本章无。
- **校准（解释误差）**：本章无；第 2 章产生第一条真机校准记录（η_compute / η_mem / T_launch）。
- **决策（提交）**：一份字段完整的纸面实验底稿（含验收清单全勾），作为后续各章实验底稿的模板。

## 本章小结

- 本书的闭环是"定义场景 → 留存预测 → 选点 → 实测 → 校准 → 留出验证 → 决策"，四条纪律（预测留痕、校准/留出不混用、重复≠泛化、结论带证据等级）贯穿全书；
- 硬件图谱：T1 RX 9070 XT（标称 640 GB/s、FP16 矩阵 195 TFLOPS，来源 AMD 官方页；ridge 点 ≈305 / ≈76 FLOP/byte），T2/T3 W7900D 48 GB 待型号确认，T4 无设备可走纯 CPU 预测；
- 测量协议最小集：容量 GiB / 带宽 GB/s、四种 ID、10 预热 + 50 测量 × 3 轮、闭环并发补位规则、校准集/留出集冻结、50 样本不支撑 p99 结论；
- 模拟器三层（解析 / DES / 校准）+ 显式边界：质量 `not_evaluated`、kernel 级黑盒、不支持的机制显式拒绝不静默忽略；
- 交付物：一份字段完整的纸面实验底稿——Qwen3-0.6B 在 T1 上的 decode 步手算（≈2.0–2.3 ms/step，memory-bound）与显存账本（KV 容量 ≈10.5 万 token）。

## 延伸阅读

- [mlsysbook.ai](https://mlsysbook.ai/) ——《Machine Learning Systems》Vol I（Harvard CS249r，Vijay Janapa Reddi 等，MIT Press，CC BY-NC-SA 4.0）：第 1 章 "The World Rushing to Build AI Systems" 与 "Iron Law of ML Systems" 是本章 0.1 的直接来源；同站 [MLSys·im 页面](https://mlsysbook.ai/#mlsys-im) 介绍本书解析层所用的解析建模框架。
- [CMU 15-442/15-642: Machine Learning Systems](https://mlsyscourse.org/)（Tianqi Chen, Zhihao Jia）：Week 1 "Why ML Systems?" 与全书主题排序（GPU 架构 → 并行化 → 显存 → LLM serving）可作为对照大纲。
- [hello-gpu](https://github.com/datawhalechina/hello-gpu)：同硬件（RX 9070 XT + ROCm）的算子/kernel 级教程，第 5 章通信校准会引用其 profile 结论。
- [hello-rocm](https://github.com/datawhalechina/hello-rocm)：ROCm 环境与部署。
- Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*（CACM 2009）——第 1 章 Roofline 图的原始文献。
- 仓库内：`curriculum/measurement-protocol.md`（0.3 节协议的完整版）、`simulator-development-plan.md`（0.4 节工具链的开发计划）、附录 A 硬件卡 / 附录 B 模型卡（发布后替换本章 §0.2 的表格）。
