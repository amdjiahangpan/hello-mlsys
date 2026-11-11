<div align="center">
  <img src="./docs/public/images/head.png" alt="Hello MLSys：用模拟预测，用真机校准，让每个系统决策有据可依" width="100%">
  <p><strong>先预测，再验证，让每个系统决策有据可依 🚀</strong></p>
</div>

<div align="center">

*以模拟为预测 · 以实测为证据 · 以校准守诚实 · 以决策为终点*

<p align="center">
  <a href="./README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
</p>

</div>

<div align="center">

<a href="https://datawhalechina.github.io/hello-mlsys/"><img alt="在线阅读" src="https://img.shields.io/badge/%F0%9F%9A%80_%E5%9C%A8%E7%BA%BF%E9%98%85%E8%AF%BB-%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="28"></a> | <a href="https://datawhalechina.github.io/hello-mlsys/curriculum/"><img alt="全书目录" src="https://img.shields.io/badge/%F0%9F%93%9A_%E5%85%A8%E4%B9%A6%E7%9B%AE%E5%BD%95-%E7%AB%A0%E8%8A%82%E5%AF%BC%E8%88%AA-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="28"></a> | <a href="https://github.com/datawhalechina/hello-rocm"><img alt="hello-rocm 环境与部署" src="https://img.shields.io/badge/%F0%9F%9B%A0%EF%B8%8F_hello--rocm-%E7%8E%AF%E5%A2%83%E4%B8%8E%E9%83%A8%E7%BD%B2-1a73e8?logo=github&logoColor=white&labelColor=1a1a1a" height="28"></a> | <a href="https://github.com/datawhalechina/hello-gpu"><img alt="hello-gpu 算子优化" src="https://img.shields.io/badge/%E2%9A%A1_hello--gpu-%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96-1a73e8?logo=github&logoColor=white&labelColor=1a1a1a" height="28"></a>

</div>

<p align="center">
  <img src="./docs/public/aup-logo.png" alt="AMD University Program" height="44">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./docs/public/rocm-logo.png" alt="AMD ROCm" height="44">
</p>

> [!CAUTION]
> ⚠️ Alpha 内测版本警告：此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提 Issue 反馈问题或建议。

&emsp;&emsp;🎓 **和新加坡国立大学（NUS）的同学一起学 MLSys！** 本项目由 NUS 的同学参与共建，也会和更多 NUS 同学一起边学边做。欢迎你加入我们，在 AMD 硬件上把 LLM 推理系统从“能跑”学到“会设计”。

&emsp;&emsp;随着大模型推理逐渐走向真实服务，模型“能跑起来”之后，问题就变成了：这张卡能放下多长的上下文、撑住多少并发？P99 延迟的拐点在哪里？换一种量化、加一张卡、换一个推理框架，收益到底有多少，代价又是什么？

&emsp;&emsp;然而，**会部署模型并不等于会设计系统**。这些问题牵涉**模型、硬件、配置、负载、SLA、质量、成本与能耗**之间的取舍，只靠反复试错，既慢又贵；只看公式和论文，又不知道预测离真机有多远。市面上的资料大多停留在“某个框架怎么用”或“某个指标是什么”，很少把“先预测、再实测、解释误差、校准模型、最后做决策”的完整链路讲透。

&emsp;&emsp;**hello-mlsys** 正是为这条路径而生。本教程面向 AMD 系列硬件：从 **Radeon RX 9070 XT、RX 7900 XTX** 等桌面显卡，到 **Ryzen AI MAX+ 395** 等端侧设备，再到 **W7900D** 多卡集群，都可以作为实验平台。每个核心实验先用模拟器 **[mlsys-simulator](https://github.com/eyuansu62/mlsys-simulator)** 预测性能、容量或设计边界，再据此挑选少量有信息价值的配置上 AMD 真机实测；解释偏差、校准模型，并在未参与拟合的新配置上验证，最后做出设计决策。全书每一章都围绕同一个学习闭环展开：

> **定义场景与约束 → 运行模拟器并留存预测 → 选择实测点 → AMD 真机验证 → 解释误差与校准 → 留出验证 → 设计决策**

&emsp;&emsp;**本教程不是推理框架的使用手册，而是希望帮助读者建立 MLSys 的核心思维：用模型做预测，用测量做证据，用校准守住诚实，用决策回答问题。任何人都可以提出 issue 或是提交 PR，共同构建维护这个项目。**

> &emsp;&emsp;**hello-rocm 教你跑起来，hello-gpu 教你把算子做快，hello-mlsys 教你使用仿真指导系统设计，并在 AMD 真机上验证和校准。** 环境安装与部署请参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm)，算子与 kernel 调优请参考 [hello-gpu](https://github.com/datawhalechina/hello-gpu)，本教程不重复这两部分内容。

> &emsp;&emsp;***学习建议：建议先读[第 0 章 协议与纸面演练](https://datawhalechina.github.io/hello-mlsys/part0-intro/chapter0/)，统一 token、GB/GiB、延迟等测量口径，并在纸面上把学习闭环走一遍；然后按「性能 → 校准 → 显存 → 量化 → 通信 → 并行 → 服务 → 框架」的顺序学习第 1 篇，再进入第 2 篇的综合设计。手边暂时没有 AMD 显卡，第 1 章的 Roofline 预测与第 3 章的显存账本都可以纯 CPU 完成，第 2 章还提供了一条 CPU 参照线作为预测入口。***

### 最新动态

- *2026.11.8:* **[教程站点](https://datawhalechina.github.io/hello-mlsys/)上线**：全书目录、章节导航、站内搜索与深浅色切换。

### 已完成章节一览

<p align="center">
  <strong>✨ 每章一个完整闭环：预测留痕 · 真机实测 · 误差归因 · 留出验证 ✨</strong><br>
  <em>支持 RX 9070 XT / RX 7900 XTX / Ryzen AI MAX+ 395 等 AMD 硬件 · 每章配套实验代码、原始测量记录与图表脚本</em><br>
  <a href="https://datawhalechina.github.io/hello-mlsys/curriculum/">全书目录</a> · <a href="https://datawhalechina.github.io/hello-mlsys/part0-intro/chapter0/">第 0 章：统一测量协议</a>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">

  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>协议 · 纸面演练</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • 教程定位、硬件图谱与统一测量协议<br>
      • 用模拟器输出填写一份字段完整的实验底稿<br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="https://datawhalechina.github.io/hello-mlsys/part0-intro/chapter0/">第 0 章 教程正文</a><br>
      • <a href="./docs/public/part0-intro/chapter0/experiment-draft.yaml">实验底稿模板</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>性能预测 · Roofline 与瓶颈</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • 算术强度、Compute-bound / Memory-bound 判定<br>
      • Prefill / Decode 宏观拆解与理论时间下界<br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter1/">第 1 章 教程正文</a><br>
      • <a href="./code/part1-modeling/chapter1">实验代码与测量记录</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>真机校准 · Profile 与留出验证</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • 真机准入检查、最小 Profile 与误差归因<br>
      • 校准集拟合与留出验证、CPU 参照线<br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter2/">第 2 章 教程正文</a><br>
      • <a href="./code/part1-modeling/chapter2">实验代码与测量记录</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>容量规划 · 权重与 KV Cache 显存账本</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • 静态权重、每 token KV 与运行时开销三本账<br>
      • 公式 / 分配器 / 设备三口径对账与可服务边界<br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter3/">第 3 章 教程正文</a><br>
      • <a href="./code/part1-modeling/chapter3">实验代码与测量记录</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>量化选型 · 精度与误差预算</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • 显存、解码速度、首 token 延迟与质量四个预算<br>
      • 低比特为什么不一定更快：反量化开销与 kernel 归因<br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter4/">第 4 章 教程正文</a><br>
      • <a href="./code/part1-modeling/chapter4">实验代码与测量记录</a><br>
    </td>
  </tr>
</table>

## 项目意义

&emsp;&emsp;什么是 MLSys？

> MLSys（Machine Learning Systems）研究如何把机器学习模型高效、可靠、经济地运行在真实硬件与真实负载上。它关心的不是单个算子有多快，而是整个系统在模型、硬件、配置、负载与服务目标之间如何取舍。

&emsp;&emsp;模型越来越大，推理越来越贵，“跑得动”之后紧接着就是“撑得住、划得来”。容量规划、量化选型、并行切分、框架选择，每一个决定都会牵动显存、延迟、吞吐与成本。然而，大多数学习资料要么停留在部署教程，要么停留在论文公式：前者告诉你怎么跑，却不告诉你为什么这样配；后者给出模型，却不告诉你它离真机有多远。

&emsp;&emsp;本项目覆盖 AMD 的多类硬件：**Radeon RX 9070 XT、RX 7900 XTX** 等桌面显卡价格亲民，学生和小型实验室也能完成真机校准；**Ryzen AI MAX+ 395** 等端侧设备把大容量统一内存带到了个人电脑上，适合端侧部署与资源受限场景；集群设计章则使用 **W7900D 48 GB 多卡**。Roofline、显存账本、α–β 通信模型与排队论在任何平台上都同样成立，换硬件只需换参数表。

&emsp;&emsp;围绕这条主线，本教程坚持三个关键词：

| 关键词 | 含义 | 你会怎么练 |
| ---- | ---- | ---- |
| **Simulator-Driven** | 先用模拟器预测，再决定测什么 | 预测先留痕，只测信息量最大的少数配置 |
| **Calibration-Honest** | 用真机裁决预测，误差必须能解释 | 拟合校准系数，在未参与拟合的配置上做留出验证 |
| **Decision-Oriented** | 每章都落到一个可量化的设计决策 | 显存可行域、量化方案、并行切分、SLA 容量、集群配置 |

&emsp;&emsp;本教程聚焦 **LLM 推理系统**；训练显存、算子调优与集群调度平台作为选读，不在本书主线范围内。

&emsp;&emsp;***我们希望让每一位学习者都能先算清楚、再测明白，用证据为每一个系统决策负责。***

## 项目受众

&emsp;&emsp;本项目适合以下学习者：

* 会用 PyTorch 跑模型，希望理解系统性能从何而来的学生和工程师；
* 想进入 MLSys / AI Infra / 推理优化方向，希望建立“预测 → 实测 → 校准”工程方法的学习者；
* 需要为推理服务做容量规划、量化选型或框架选型，却缺少系统化判断依据的开发者；
* 希望了解 AMD GPU / ROCm 生态下 LLM 推理系统设计的学习者。

> &emsp;&emsp;你不需要有算子开发经验，但最好具备 Python、PyTorch、Linux 基础与英文文档阅读能力。

## 项目规划及进展

&emsp;&emsp;全书沿着「**建立口径 → 建模与校准 → 与 AI 一起设计 → 选修拓展**」的主线组织，共 4 篇正文与 6 个附录。每一章都是一个完整的“预测 → 实测 → 校准 → 决策”闭环，后面的章节会复用前面校准过的模型，建议按顺序学习。

### 章节与实验覆盖

> ✅ 表示该章包含对应平台上的实验，🚧 表示该平台正在适配中。

| 篇 / 章 | 核心实验 / 主题 | 模拟器 | RX 9070 XT | W7900D | Ryzen AI MAX+ 395 | 状态 |
| ---- | ---- | :----: | :----: | :----: | :----: | :----: |
| [Part 0 入门](https://datawhalechina.github.io/hello-mlsys/part0-intro/chapter0/) | 定位、硬件图谱、统一测量协议、模拟器工具链 | ✅ | | | | 🚧 |
| [Ch 1](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter1/) | 使用模拟器估算时间、Roofline 与瓶颈 | ✅ | | | | 🚧 |
| [Ch 2](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter2/) | 第一次跑通 Profile，并使用实际样例校准 | ✅ | ✅ | ✅ | 🚧 | 🚧 |
| [Ch 3](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter3/) | 显存占用估算：模型权重与 KV Cache | ✅ | ✅ | ✅ | 🚧 | 🚧 |
| [Ch 4](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter4/) | 精度、量化方法与误差预算 | ✅ | ✅ | ✅ | 🚧 | 🚧 |
| [Ch 5](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter5/) | 通信代价模型与 Scale-in / Scale-up | ✅ | | | | 🚧 |
| [Ch 6](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter6/) | 模型并行策略与显存 / 算力拆解 | ✅ | ✅ | ✅ | 🚧 | 🚧 |
| [Ch 7](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter7/) | 服务、排队与 SLA，负载生成与压测 | ✅ | ✅ | ✅ | 🚧 | 🚧 |
| [Ch 8](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter8/) | 推理框架的系统取舍、Profile 方式与性能分析 | ✅ | ✅ | ✅ | 🚧 | 🚧 |
| [Ch 9.1](https://datawhalechina.github.io/hello-mlsys/part2-ai-design/chapter9-1/) | 与 AI 一起实战：端侧部署助手 | | ✅ | ✅ | 🚧 | 🚧 |
| [Ch 9.2](https://datawhalechina.github.io/hello-mlsys/part2-ai-design/chapter9-2/) | 与 AI 一起实战：单机服务满足 SLA | | ✅ | ✅ | 🚧 | 🚧 |
| [Ch 9.3](https://datawhalechina.github.io/hello-mlsys/part2-ai-design/chapter9-3/) | 与 AI 一起实战：多卡长上下文、多 Subagent 请求处理 | ✅ | | ✅ | | 🚧 |
| [Part 3](https://datawhalechina.github.io/hello-mlsys/part3-electives/) | 选修与案例库：优先选修实验与扩展案例卡 | | ✅ | ✅ | 🚧 | 🚧 |
| [附录](https://datawhalechina.github.io/hello-mlsys/curriculum/) | 硬件参数、模型卡、术语表、参考资源、社区共建、常见失败模式与排查清单 | | | | | 🚧 |

### 项目结构

```
hello-mlsys/
├── docs/                      # VitePress 教程站点源文件
│   ├── part0-intro/           # 第 0 篇：入门
│   ├── part1-modeling/        # 第 1 篇：建模与校准
│   ├── part2-ai-design/       # 第 2 篇：与 AI 一起设计
│   ├── part3-electives/       # 第 3 篇：选修与案例库
│   ├── appendix/              # 附录 A–F
│   ├── curriculum/            # 全书目录（由大纲生成）
│   └── .vitepress/            # 站点配置与全书大纲（outline.mjs）
├── code/                      # 各章实验代码、测量记录与图表脚本
├── notebooks/                 # Notebook 动手学习
└── scripts/                   # 环境配置、构建与检查脚本
```

## 目录

| 章节名 | 简介 | 状态 |
| ---- | ---- | ---- |
| **第 0 篇：入门** | | |
| [第 0 章 协议与纸面演练](./docs/part0-intro/chapter0/index.md) | 教程定位、硬件图谱、统一测量协议、模拟器工具链与能力边界 | 🚧 |
| **第 1 篇：建模与校准** | | |
| [第 1 章 模拟器估算、Roofline 与瓶颈](./docs/part1-modeling/chapter1/index.md) | Roofline、算术强度、Compute-bound / Memory-bound 判定、计算时间量级估算 | 🚧 |
| [第 2 章 第一次真机 Profile 与性能校准](./docs/part1-modeling/chapter2/index.md) | 真机准入检查、最小 Profile 跑通、误差归因、校准集与留出验证 | 🚧 |
| [第 3 章 显存占用估算：模型权重与 KV Cache](./docs/part1-modeling/chapter3/index.md) | 权重显存、KV Cache、激活与碎片开销、最大可服务上下文与并发 | 🚧 |
| [第 4 章 精度、量化方法与误差预算](./docs/part1-modeling/chapter4/index.md) | 数值格式、精度 / 延迟 / 显存三维权衡、误差预算分配 | 🚧 |
| [第 5 章 通信代价模型与 Scale-in / Scale-up](./docs/part1-modeling/chapter5/index.md) | 节点内与节点间互联、集合通信原语、α–β 代价模型 | 🚧 |
| [第 6 章 模型并行策略与显存 / 算力拆解](./docs/part1-modeling/chapter6/index.md) | DP / FSDP、TP、PP、切分收益被通信吃掉的判定条件 | 🚧 |
| [第 7 章 服务、排队与 SLA](./docs/part1-modeling/chapter7/index.md) | TTFT、TPOT / ITL、分位数延迟、并发与吞吐、排队拐点、负载生成与压测 | 🚧 |
| [第 8 章 推理框架的系统取舍、Profile 与性能分析](./docs/part1-modeling/chapter8/index.md) | 框架级取舍、框架级 Profile、配置空间预测与失败模式 | 🚧 |
| **第 2 篇：与 AI 一起设计** | | |
| [第 9.1 章 端侧部署助手](./docs/part2-ai-design/chapter9-1/index.md) | 资源受限场景下的模型、精度、上下文与 KV 配置选型 | 🚧 |
| [第 9.2 章 单机服务满足 SLA](./docs/part2-ai-design/chapter9-2/index.md) | 单卡容量规划：显存上界与排队上界、实测拐点与留出验证 | 🚧 |
| [第 9.3 章 负载启发的推理集群设计](./docs/part2-ai-design/chapter9-3/index.md) | 冻结的 deep-research 轨迹上以总墙钟最短为目标：32B/7B 两档模型协作与多种并行策略，先模拟筛选配置，再在 W7900D 48 G 多卡集群实测 | 🚧 |
| **第 3 篇：选修与案例库** | | |
| [选修与案例库](./docs/part3-electives/index.md) | 优先选修实验与扩展案例卡 | 🚧 |
| **附录** | | |
| [附录 A 硬件参数](./docs/appendix/appendix-a-hardware/index.md) | 待补充 | 🚧 |
| [附录 B 模型卡](./docs/appendix/appendix-b-model-cards/index.md) | 待补充 | 🚧 |
| [附录 C 术语表](./docs/appendix/appendix-c-glossary/index.md) | 待补充 | 🚧 |
| [附录 D 参考资源](./docs/appendix/appendix-d-references/index.md) | 待补充 | 🚧 |
| [附录 E 社区共建](./docs/appendix/appendix-e-community/index.md) | 待补充 | 🚧 |
| [附录 F 常见失败模式与排查清单](./docs/appendix/appendix-f-troubleshooting/index.md) | 待补充 | 🚧 |

## 贡献者名单

### 核心贡献者

- 陆遥-指导教师（新加坡国立大学计算机学院助理教授）
- yuzhou-项目负责人（上海科技大学博士在读）
- [潘嘉航-项目负责人](https://github.com/amdjiahangpan)（AMD 软件工程师）
- [秦博文-项目成员](https://github.com/eyuansu62)（新加坡国立大学博士在读，[mlsys-simulator](https://github.com/eyuansu62/mlsys-simulator) 模拟器搭建与校准）
- [刘伟鸿-项目成员](https://github.com/Weihong-Liu)（Datawhale 成员）
- yanhui-项目成员（新加坡国立大学 MComp 硕士）
- jiyan-项目成员（新加坡国立大学 MComp 硕士）

> 注：欢迎更多贡献者加入！

### 其他

- 如果有任何想法可以联系我们，也欢迎大家多多提出 issue
- 特别感谢以下为教程做出贡献的同学！
- 感谢 AMD University Program 对本项目的支持！

<div align="center">
  <a href="https://github.com/datawhalechina/hello-mlsys/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-mlsys" alt="Hello MLSys 项目贡献者" />
  </a>
</div>

## 参与贡献

&emsp;&emsp;我们欢迎所有形式的贡献！无论是：

* 完善或新增教程章节
* 修复错误与 Bug
* 在其他 AMD GPU 上复跑实验、补充实测与校准数据
* 提出建议与想法

&emsp;&emsp;参与前请先阅读 **[贡献指南 CONTRIBUTING.md](./CONTRIBUTING.md)**，其中说明了内容放置位置、章节与实验记录规范、本地预览方式与 PR 流程。

- 如果你发现了一些问题，可以提 Issue 进行反馈；
- 如果你想参与贡献本项目，可以提 Pull Request，我们会尽快 review。

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

---

<div align="center">

**让我们一起先预测、再验证，用证据设计每一个系统！** 💪

Made with ❤️ by the hello-mlsys community

</div>
