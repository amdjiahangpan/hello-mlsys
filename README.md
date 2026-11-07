<h1 align="center"> Hello MLSys（⚠️ Alpha内测版） </h1>

> [!CAUTION]
> ⚠️ Alpha内测版本警告：此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。

随着大模型推理逐渐走向真实服务，模型“能跑起来”之后，问题就变成了：这张卡能放下多长的上下文、撑住多少并发？P99 延迟的拐点在哪里？换一种量化、加一张卡、换一个推理框架，收益到底有多少，代价又是什么？这些问题牵涉**模型、硬件、配置、负载、SLA、质量、成本与能耗**之间的取舍，只靠反复试错，既慢又贵。

**hello-mlsys** 希望换一种学法：**先预测，再验证**。每个核心实验先用模拟器预测性能、容量或设计边界，再据此挑选少量有信息价值的配置上 AMD 真机实测；解释偏差、校准模型，并在未参与拟合的新配置上验证，最后做出设计决策。全书每一章都围绕同一个学习闭环展开：

> **定义场景与约束 → 运行模拟器并留存预测 → 选择实测点 → AMD 真机验证 → 解释误差与校准 → 留出验证 → 设计决策**

本教程不是推理框架的使用手册，而是希望帮助读者建立 MLSys 的核心思维：**用模型做预测，用测量做证据，用校准守住诚实，用决策回答问题。**

> hello-rocm 教你跑起来，hello-gpu 教你把算子做快，hello-mlsys 教你用模拟指导系统设计，并在 AMD 真机上验证和校准。环境安装与部署请参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm)，算子与 kernel 调优请参考 [hello-gpu](https://github.com/datawhalechina/hello-gpu)，本教程不重复这两部分内容。

本教程聚焦 **LLM 推理系统**；训练显存、算子调优与集群调度平台作为选读。

## 项目受众

- 会用 PyTorch 跑模型，希望理解系统性能从何而来的学生和工程师；
- 想进入 MLSys / AI Infra / 推理优化方向，希望建立“预测 → 实测 → 校准”工程方法的学习者；
- 需要为推理服务做容量规划、量化选型或框架选型，却缺少系统化判断依据的开发者；
- 希望了解 AMD GPU / ROCm 生态下 LLM 推理系统设计的学习者。

> 你不需要有算子开发经验，但最好具备 Python、PyTorch、Linux 基础与英文文档阅读能力。

## 在线阅读

🚧 待上线

## 本地预览与构建

站点使用 VitePress 默认主题，直接把 `docs/` 中的 Markdown 渲染为教程页面，提供全书目录、章节导航、站内搜索与深浅色切换。

使用 Node.js 22.12 或更新版本，在仓库根目录运行：

```bash
npm install
npm run docs:dev
```

打开终端显示的本地地址（默认 `http://localhost:5173/hello-mlsys/`）。修改 Markdown 后，页面会自动刷新。

构建与检查：

```bash
npm run docs:check-outline
npm run docs:build
npm run docs:preview
```

静态页面生成到 `docs/.vitepress/dist/`。`docs/.vitepress/outline.mjs` 是章节信息的统一入口，`docs:sync-outline` 会同步 README、全书目录页与章节元数据，保留已有正文。`docs/curriculum/index.md` 是生成文件，请通过大纲修改。

### GitHub 自动构建与发布

仓库的 `.github/workflows/deploy.yml` 已配置自动流程：

- 向 `main` 推送提交（包括合并 PR）：自动安装依赖、构建并发布到 GitHub Pages。
- 向 `main` 提交 PR：检查站点能否构建，合并后再发布。
- 在 Actions 中手动运行工作流：选择 `main` 可以重新构建并发布；其他分支只构建。

首次发布需要**仓库管理员**在 Settings → Pages 中将 Build and deployment 的 Source 设为 **GitHub Actions**。如果该选项不可用，请管理员确认当前私有仓库的 Pages 使用权限与站点访问设置。

管理员完成这一次设置、代码合并到 `main` 后，后续只需修改 Markdown、提交并推送，网页就会随工作流更新。发布状态可在 Actions 的 **Deploy VitePress site to Pages** 中查看，成功后的网址以部署结果为准。

构建使用 Node.js 22 和 `npm ci`，请将 `package-lock.json` 一并提交；修改依赖时运行 `npm install` 更新锁文件。当前站点路径由 `docs/.vitepress/config.mts` 中的 `base: '/hello-mlsys/'` 配置。

## 项目结构

```text
hello-mlsys/
├── docs/                      # VitePress 教程站点源文件
│   ├── part0-intro/           # 第 0 篇：入门
│   ├── part1-modeling/        # 第 1 篇：建模与校准
│   ├── part2-ai-design/       # 第 2 篇：与 AI 一起设计
│   ├── part3-electives/       # 第 3 篇：选修与案例库
│   ├── appendix/              # 附录
│   └── .vitepress/            # 站点配置与全书大纲（outline.mjs）
├── code/                      # 各章实验代码、配置与测量记录
├── notebooks/                 # Notebook 动手学习
└── scripts/                   # 环境配置、构建与检查脚本
```

## 目录

> 章节由 `docs/.vitepress/outline.mjs` 统一维护，修改后运行 `npm run docs:sync-outline` 即可同步 README 目录、站点导航与章节骨架。

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

> 注：欢迎更多贡献者加入！

### 其他

- 如果有任何想法可以联系我们，也欢迎大家多多提出 issue
- 特别感谢以下为教程做出贡献的同学！
- 感谢 AMD University Program 对本项目的支持！

<a href="https://github.com/amdjiahangpan/hello-mlsys/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=amdjiahangpan/hello-mlsys" alt="Hello MLSys 项目贡献者" />
</a>

## 参与贡献

我们欢迎所有形式的贡献！无论是：

- 完善或新增教程章节
- 修复错误与 Bug
- 在其他 AMD GPU 上复跑实验、补充实测与校准数据
- 提出建议与想法

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
