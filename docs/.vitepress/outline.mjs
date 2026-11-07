// 全书大纲：站点导航、侧边栏、全书目录、章节骨架与 README 目录均由此文件生成。
// 修改后运行 `npm run docs:sync-outline` 同步。

export const repoUrl = 'https://github.com/amdjiahangpan/hello-mlsys'

export const parts = [
  {
    prefix: '/part0-intro/',
    navText: '入门',
    title: '入门：协议与纸面演练',
    readmeTitle: '第 0 篇：入门',
    chapters: [
      {
        path: '/part0-intro/chapter0/',
        source: 'docs/part0-intro/chapter0/index.md',
        code: 'code/part0-intro/chapter0',
        label: '第 0 章',
        number: '0',
        title: '协议与纸面演练',
        summary: '教程定位、硬件图谱、统一测量协议、模拟器工具链与能力边界',
        status: '🚧',
        lead: '本章说明本教程的定位与学习路径，建立贯穿全书的统一测量协议，并认识模拟器工具链能做什么、不能做什么。读完后，你应该能用模拟器输出填写一份字段完整的纸面实验底稿。',
        sections: [
          ['教程定位与学习路径', '说明本教程与 hello-rocm、hello-gpu 的分工，以及“预测 → 实测 → 校准 → 决策”的学习闭环。'],
          ['AMD GPU 硬件图谱', '整理后续实验需要的算力、带宽、显存与互联参数，并注明参数来源。'],
          ['统一测量协议', '统一 token、GB/GiB、延迟与功耗口径，约定预热、重复次数、方差记录，以及校准集与留出集的划分。'],
          ['模拟器工具链与能力边界', '认识模拟器覆盖的性能、显存、排队与通信建模范围，区分已支持与待补齐的能力。'],
          ['纸面实验底稿', '用模拟器输出填写一份实验底稿，确认硬件版本、软件版本、预热次数、重复次数与留出配置等字段完整。']
        ]
      }
    ]
  },
  {
    prefix: '/part1-modeling/',
    navText: '建模与校准',
    title: '建模与校准',
    readmeTitle: '第 1 篇：建模与校准',
    chapters: [
      {
        path: '/part1-modeling/chapter1/',
        source: 'docs/part1-modeling/chapter1/index.md',
        code: 'code/part1-modeling/chapter1',
        label: '第 1 章',
        number: '1',
        title: '模拟器估算、Roofline 与瓶颈',
        summary: 'Roofline、算术强度、Compute-bound / Memory-bound 判定、计算时间量级估算',
        status: '🚧',
        lead: '本章建立性能预测的第一层直觉：一个算子在给定硬件上的理论上限是多少，瓶颈落在计算还是访存。读完后，你应该能先用模拟器给出预测，再带着预测去看真机数据。',
        sections: [
          ['Roofline 模型', '理解 Attainable FLOPS = min(Peak FLOPS, AI × Peak Bandwidth) 的含义与适用边界。'],
          ['算术强度与瓶颈判定', '计算算子的算术强度，判断它是 Compute-bound 还是 Memory-bound。'],
          ['计算时间量级估算', '从工作量与硬件峰值估算执行时间的数量级。'],
          ['模拟器实验：预测典型算子的瓶颈与时间', '用模拟器预测矩阵乘、LayerNorm 与 FlashAttention 的瓶颈类型和预测时间，并用一个“看起来像 Compute-bound、实际受带宽限制”的算子校准直觉。']
        ]
      },
      {
        path: '/part1-modeling/chapter2/',
        source: 'docs/part1-modeling/chapter2/index.md',
        code: 'code/part1-modeling/chapter2',
        label: '第 2 章',
        number: '2',
        title: '第一次真机 Profile 与性能校准',
        summary: '真机准入检查、最小 Profile 跑通、误差归因、校准集与留出验证',
        status: '🚧',
        lead: '本章第一次把模拟器预测带上真机：先跑通最小 Profile，再用校准集拟合修正系数，最后在留出配置上检验修正能否泛化。读完后，你应该能完成一次完整的“预测 → 实测 → 校准 → 留出验证”。',
        sections: [
          ['真机准入检查', '确认硬件、驱动与软件版本满足实验要求，并记录实验环境。'],
          ['第一次最小 Profile', '用 rocprof-compute 跑通一个简单 kernel，不追求全面指标，只确认工具链可用。'],
          ['校准集：预测与实测对比', '选取若干算子，对比模拟器预测与实测结果，计算误差并拟合修正系数。'],
          ['误差归因', '检查降频、内存碎片、软件版本与模拟器假设不一致等误差来源，并逐项记录。'],
          ['留出验证', '在未参与校准的算子上验证“预测 → 修正 → 再预测”的误差是否在可接受范围。'],
          ['端到端模型运行与 Profile', '跑通一个小规模 LLM 的端到端推理与 Profile。']
        ]
      },
      {
        path: '/part1-modeling/chapter3/',
        source: 'docs/part1-modeling/chapter3/index.md',
        code: 'code/part1-modeling/chapter3',
        label: '第 3 章',
        number: '3',
        title: '显存占用估算：模型权重与 KV Cache',
        summary: '权重显存、KV Cache、激活与碎片开销、最大可服务上下文与并发',
        status: '🚧',
        lead: '本章回答“什么能放得下”：把显存拆成权重、KV Cache 与运行时开销三本账，先用公式手算，再与模拟器和真机读数对答案。',
        sections: [
          ['权重显存', '权重显存 = 参数量 × 每参数字节数。'],
          ['KV Cache 显存', '每 token KV 字节数 = 2 × 层数 × KV 头数 × 头维度 × 每元素字节数，理解 MHA 与 GQA 的差异。'],
          ['激活与碎片开销', '区分张量占用、allocated / reserved 与设备总占用。'],
          ['模拟器实验：最大可服务上下文与并发', '给定模型配置与序列长度，预测权重显存、每 token KV 显存，以及目标卡上的最大可服务上下文与并发数。'],
          ['真机验证：权重与 KV Cache', '对比加载模型前后的显存读数，观察推理框架的 KV Cache 预留行为。'],
          ['改变序列长度与并发', '调整最大序列长度或并发数，测量 KV Cache 增长并与公式预测对比。']
        ]
      },
      {
        path: '/part1-modeling/chapter4/',
        source: 'docs/part1-modeling/chapter4/index.md',
        code: 'code/part1-modeling/chapter4',
        label: '第 4 章',
        number: '4',
        title: '精度、量化方法与误差预算',
        summary: '数值格式、精度 / 延迟 / 显存三维权衡、误差预算分配',
        status: '🚧',
        lead: '本章把量化当成一个多约束的设计问题：显存、延迟与质量三者同时有预算，哪个约束最先被撑破？读完后，你应该理解“显存减半”与“延迟下降”并不是一回事。',
        sections: [
          ['数值格式', '比较 FP16、BF16、FP8、INT8、INT4 的表示范围与精度差异。'],
          ['精度、延迟与显存的三维权衡', '理解量化如何同时影响三个维度，以及反量化开销从何而来。'],
          ['误差预算', '在质量底线约束下分配误差预算。'],
          ['模拟器实验：多约束下的量化方案筛选', '给定显存、延迟与精度损失约束，预测不同量化方案能否同时满足，并指出最先被撑破的约束。'],
          ['真机验证：显存、延迟与质量', '对同一模型的不同精度版本测量显存占用、延迟中位数与小评测集上的质量。'],
          ['低比特为什么不一定更快', '用实测数据解释量化收益的边界。']
        ]
      },
      {
        path: '/part1-modeling/chapter5/',
        source: 'docs/part1-modeling/chapter5/index.md',
        code: 'code/part1-modeling/chapter5',
        label: '第 5 章',
        number: '5',
        title: '通信代价模型与 Scale-in / Scale-up',
        summary: '节点内与节点间互联、集合通信原语、α–β 代价模型',
        status: '🚧',
        lead: '本章建立通信的一阶代价模型 T ≈ α + β × Data，并区分“行为验证”与“性能验证”。读完后，你应该能判断一次集合通信处在延迟主导区还是带宽主导区。',
        sections: [
          ['节点内与节点间互联', '比较 PCIe、xGMI 等节点内链路与 InfiniBand、RoCEv2 等节点间网络的带宽与延迟量级。'],
          ['集合通信原语', '理解 All-Reduce、All-Gather、All-to-All 的数据流动方式。'],
          ['α–β 代价模型', '用启动开销与带宽项刻画通信时间。'],
          ['模拟器实验：不同拓扑下的 All-Reduce', '预测不同拓扑下 All-Reduce 时间随数据量的变化，标出延迟主导区与带宽主导区的分界点。'],
          ['真机实验：主机与设备间带宽', '测量 H2D / D2H 带宽，作为 PCIe 路径的实测参考。'],
          ['行为验证与性能验证', '观察集合通信 API 与拓扑感知逻辑的真实行为，明确哪些实测值不能代表多卡性能。']
        ]
      },
      {
        path: '/part1-modeling/chapter6/',
        source: 'docs/part1-modeling/chapter6/index.md',
        code: 'code/part1-modeling/chapter6',
        label: '第 6 章',
        number: '6',
        title: '模型并行策略与显存 / 算力拆解',
        summary: 'DP / FSDP、TP、PP、切分收益被通信吃掉的判定条件',
        status: '🚧',
        lead: '本章以模拟器为主角，面对一个单卡放不下的模型做真实的切分决策。读完后，你应该能用每卡显存与通信占比两个指标比较不同并行方案。',
        sections: [
          ['DP / FSDP', '理解数据并行的显存冗余与梯度同步。'],
          ['TP', '理解层内切分与频繁通信。'],
          ['PP', '理解层间切分与流水线气泡。'],
          ['切分收益何时被通信吃掉', '给出切分收益与通信代价的判定条件。'],
          ['模拟器实验：单卡放不下的模型如何切分', '分别用 DP、TP、PP 拆分同一模型，比较每卡显存占用与通信时间占比。'],
          ['留出验证', '换一个模型规模，检查之前的切分结论是否仍然成立。'],
          ['真机实验：多 rank 代码路径', '在单卡上启动多个 rank，观察切分后的显存分片与集合通信调用，区分行为验证与性能验证。']
        ]
      },
      {
        path: '/part1-modeling/chapter7/',
        source: 'docs/part1-modeling/chapter7/index.md',
        code: 'code/part1-modeling/chapter7',
        label: '第 7 章',
        number: '7',
        title: '服务、排队与 SLA',
        summary: 'TTFT、TPOT / ITL、分位数延迟、并发与吞吐、排队拐点、负载生成与压测',
        status: '🚧',
        lead: '本章从单次推理走向在线服务：请求会排队，延迟有分布，SLA 由尾部决定。读完后，你应该能预测并实测出“SLA 即将被突破”的并发水位。',
        sections: [
          ['服务延迟指标', '定义 TTFT、TPOT / ITL 与端到端延迟。'],
          ['分位数与尾延迟', '理解 P50、P95、P99 在 SLA 中的作用。'],
          ['并发、吞吐与排队拐点', '理解并发数与吞吐的关系，以及排队拐点的成因；区分 batch 大小与在线用户数。'],
          ['模拟器实验：P99 随并发的变化', '给定服务时间分布与到达率，预测 P99 随并发增长的曲线，标出 SLA 水位。'],
          ['真机压测：扫描并发', '启动推理服务，扫描并发并输出 TTFT、TPOT 与 P99 曲线。'],
          ['拐点对比与误差来源', '对比预测与实测拐点，分析 KV Cache 抢占、调度延迟与算力饱和等误差来源。']
        ]
      },
      {
        path: '/part1-modeling/chapter8/',
        source: 'docs/part1-modeling/chapter8/index.md',
        code: 'code/part1-modeling/chapter8',
        label: '第 8 章',
        number: '8',
        title: '推理框架的系统取舍、Profile 与性能分析',
        summary: '框架级取舍、框架级 Profile、配置空间预测与失败模式',
        status: '🚧',
        lead: '本章是第 1 篇的收口章：用前面建立的 Roofline、显存、量化与排队模型，解释不同推理框架在同一场景下的差异，并据此做出框架与配置选择。',
        sections: [
          ['框架级系统取舍', '理解显存碎片与吞吐、调度开销与批处理收益之间的权衡。'],
          ['框架级 Profile 方法', '用 Profiler 与框架内置指标观察 decode 阶段的 kernel 时间占比。'],
          ['模拟器实验：配置空间与失败模式', '给定 SLA 目标与框架配置空间，预测最可能满足 SLA 的配置，以及显存先爆还是延迟先爆。'],
          ['真机对比：不同推理框架', '在同一模型与同一压测条件下对比不同框架的 P99 与吞吐，注明模型格式、精度与调度的不等价项。'],
          ['串起前面的模型', '用显存、量化、Roofline 与排队模型解释框架差异。']
        ]
      }
    ]
  },
  {
    prefix: '/part2-ai-design/',
    navText: '与 AI 一起设计',
    title: '与 AI 一起设计',
    readmeTitle: '第 2 篇：与 AI 一起设计',
    chapters: [
      {
        path: '/part2-ai-design/chapter9-1/',
        source: 'docs/part2-ai-design/chapter9-1/index.md',
        code: 'code/part2-ai-design/chapter9-1',
        label: '第 9.1 章',
        number: '9.1',
        title: '端侧部署助手',
        summary: '资源受限场景下的模型、精度、上下文与 KV 配置选型',
        status: '🚧',
        lead: '本章在资源受限的端侧约束下，与 AI 一起完成一次部署选型：先用模拟器筛选候选方案，再选少量配置实测，把采纳、修改或否决的理由写进决策日志。',
        sections: [
          ['端侧约束与候选方案', '明确显存、首 token 延迟与解码速度约束，列出模型与量化候选。'],
          ['模拟器筛选', '预测哪些组合能同时满足约束，并指出最紧的约束。'],
          ['真机验证', '对选中的配置测量显存、TTFT 与解码速度，与预测的可服务性对比。'],
          ['设计决策与决策日志', '在“能放下但更慢”与“更小模型但更快”之间做出取舍并记录理由。']
        ]
      },
      {
        path: '/part2-ai-design/chapter9-2/',
        source: 'docs/part2-ai-design/chapter9-2/index.md',
        code: 'code/part2-ai-design/chapter9-2',
        label: '第 9.2 章',
        number: '9.2',
        title: '单机服务满足 SLA',
        summary: '单卡容量规划：显存上界与排队上界、实测拐点与留出验证',
        status: '🚧',
        lead: '本章综合运用量化、排队与推理框架三章的模型，在单机上通过框架参数、量化与并发限制满足给定 SLA。',
        sections: [
          ['SLA 目标与容量上界', '用显存模型预测可开的最大 KV Cache，用排队模型预测目标 SLA 下的最大安全并发，两者取小。'],
          ['实测拐点', '扫描并发，找到 P99 刚好不超标的并发点，并与预测上界对比。'],
          ['误差归因', '分析显存碎片、调度开销与功耗降频等偏差来源。'],
          ['留出验证', '更换 prompt 长度分布，检查之前选定的并发限制是否仍然安全。']
        ]
      },
      {
        path: '/part2-ai-design/chapter9-3/',
        source: 'docs/part2-ai-design/chapter9-3/index.md',
        code: 'code/part2-ai-design/chapter9-3',
        label: '第 9.3 章',
        number: '9.3',
        title: '负载启发的推理集群设计',
        summary: '冻结的 deep-research 轨迹上以总墙钟最短为目标：32B/7B 两档模型协作与多种并行策略，先模拟筛选配置，再在 W7900D 48 G 多卡集群实测',
        status: '🚧',
        lead: '本章是全书的结营章：给定一条冻结的 claudecode deep-research 轨迹（50+ subagent、32B/7B 两档请求），在 W7900D 48 G 多卡集群上用多种并行策略与多档模型协作，使总墙钟最短。先用模拟器完成配置筛选并留存预测，再实测验证，明确区分实测与外推。',
        sections: [
          ['冻结轨迹与约束', '固定的 deep-research 轨迹：50+ subagent、多控制节点 RAG、32B/7B 两档请求；请求内容冻结不可改，infra 策略是唯一决策变量。'],
          ['候选 infra 配置', 'W7900D 48 G 多卡集群（≥4 卡）的部署参考：32B 的并行/精度候选与 7B 的副本数，显存余量、并行通信开销与排队长度的取舍。'],
          ['墙钟目标与调度', '以总墙钟最短为目标的 DAG 调度：subagent 依赖与到达顺序、优先级/抢占、批处理与 KV 预算。'],
          ['实测分层与回退', '主线为 W7900D 集群的端到端闭环；无集群资源时 9070 XT 16 GB 只跑 7B/4B 档缩减轨迹。'],
          ['验证边界与外推', '留出配置上先预测后实测；设计文档中区分实测验证与模拟外推。']
        ]
      }
    ]
  },
  {
    prefix: '/part3-electives/',
    navText: '选修与案例库',
    title: '选修与案例库',
    readmeTitle: '第 3 篇：选修与案例库',
    chapters: [
      {
        path: '/part3-electives/',
        source: 'docs/part3-electives/index.md',
        code: 'code/part3-electives',
        label: '',
        number: '',
        title: '选修与案例库',
        summary: '优先选修实验与扩展案例卡',
        status: '🚧',
        lead: '必修章节之后的优先选修与扩展案例卡。案例卡统一按“场景 → 预测 → 验证 → 复盘”组织：学生页先给问题、约束与模拟入口，实测答案和原因放复盘页。',
        sections: [
          ['C05（优先选修）：KV 搬运 vs 重算', '已有解析计算器；补 AMD 实际传输与重算参数、路径分解和边界验证。'],
          ['案例卡清单', '统一格式“场景 → 预测 → 验证 → 复盘”：02 缓存层次、04 双卡 MTP、05 投机解码、06 MoE 路由、09 MLA/CSA/HCA、11 存储后端、双机互联。'],
          ['移出必修', '07 MoE kernel 参数调优移出必修，关联 hello-gpu 作为选读。']
        ]
      }
    ]
  },
  {
    prefix: '/appendix/',
    navText: '附录',
    title: '附录',
    readmeTitle: '附录',
    chapters: [
      ['a', 'hardware', '硬件参数'],
      ['b', 'model-cards', '模型卡'],
      ['c', 'glossary', '术语表'],
      ['d', 'references', '参考资源'],
      ['e', 'community', '社区共建'],
      ['f', 'troubleshooting', '常见失败模式与排查清单']
    ].map(([letter, slug, title]) => ({
      path: `/appendix/appendix-${letter}-${slug}/`,
      source: `docs/appendix/appendix-${letter}-${slug}/index.md`,
      label: `附录 ${letter.toUpperCase()}`,
      number: '',
      title,
      summary: '待补充',
      status: '🚧',
      lead: '待补充。',
      sections: []
    }))
  }
]

export const chapters = parts.flatMap((part) => part.chapters.map((chapter) => ({ ...chapter, part })))

export function chapterHeading(chapter) {
  return chapter.label ? `${chapter.label} ${chapter.title}` : chapter.title
}

export const navItems = [
  { text: '首页', link: '/' },
  { text: '全书目录', link: '/curriculum/' },
  {
    text: '阅读教程',
    items: parts.map((part) => ({
      text: part.navText,
      link: part.chapters[0].path,
    })),
  },
]

export const sidebar = parts.map((part) => ({
  text: part.readmeTitle,
  collapsed: false,
  items: part.chapters.map((chapter) => ({
    text: chapterHeading(chapter),
    link: chapter.path,
  })),
}))
