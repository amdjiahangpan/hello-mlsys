---
title: 全书目录
description: Hello MLSys 的学习路线、全部章节与附录。
outline: [2, 2]
editLink: false
prev: false
next: false
---

<!-- 此页由 docs/.vitepress/sync-outline.mjs 生成，请在 outline.mjs 中维护章节。 -->

# 全书目录

从入门开始，依次学习建模与校准，再完成与 AI 一起设计的综合实践。选修案例和附录可按需查阅。

::: info 阅读说明
教程处于 Alpha 阶段，部分章节仍为大纲或待补充内容，请以各章现有正文与验证记录为准。
:::

## 第 0 篇：入门

- [第 0 章 协议与纸面演练](/part0-intro/chapter0/) — 教程定位、硬件图谱、统一测量协议、模拟器工具链与能力边界

## 第 1 篇：建模与校准

- [第 1 章 模拟器估算、Roofline 与瓶颈](/part1-modeling/chapter1/) — Roofline、算术强度、Compute-bound / Memory-bound 判定、计算时间量级估算
- [第 2 章 第一次真机 Profile 与性能校准](/part1-modeling/chapter2/) — 真机准入检查、最小 Profile 跑通、误差归因、校准集与留出验证
- [第 3 章 显存占用估算：模型权重与 KV Cache](/part1-modeling/chapter3/) — 权重显存、KV Cache、激活与碎片开销、最大可服务上下文与并发
- [第 4 章 精度、量化方法与误差预算](/part1-modeling/chapter4/) — 数值格式、精度 / 延迟 / 显存三维权衡、误差预算分配
- [第 5 章 通信代价模型与 Scale-in / Scale-up](/part1-modeling/chapter5/) — 节点内与节点间互联、集合通信原语、α–β 代价模型
- [第 6 章 模型并行策略与显存 / 算力拆解](/part1-modeling/chapter6/) — DP / FSDP、TP、PP、切分收益被通信吃掉的判定条件
- [第 7 章 服务、排队与 SLA](/part1-modeling/chapter7/) — TTFT、TPOT / ITL、分位数延迟、并发与吞吐、排队拐点、负载生成与压测
- [第 8 章 推理框架的系统取舍、Profile 与性能分析](/part1-modeling/chapter8/) — 框架级取舍、框架级 Profile、配置空间预测与失败模式

## 第 2 篇：与 AI 一起设计

- [第 9.1 章 端侧部署助手](/part2-ai-design/chapter9-1/) — 资源受限场景下的模型、精度、上下文与 KV 配置选型
- [第 9.2 章 单机服务满足 SLA](/part2-ai-design/chapter9-2/) — 单卡容量规划：显存上界与排队上界、实测拐点与留出验证
- [第 9.3 章 负载启发的推理集群设计](/part2-ai-design/chapter9-3/) — 冻结的 deep-research 轨迹上以总墙钟最短为目标：32B/7B 两档模型协作与多种并行策略，先模拟筛选配置，再在 W7900D 48 G 多卡集群实测

## 第 3 篇：选修与案例库

- [选修与案例库](/part3-electives/) — 优先选修实验与扩展案例卡

## 附录

- [附录 A 硬件参数](/appendix/appendix-a-hardware/) — 待补充
- [附录 B 模型卡](/appendix/appendix-b-model-cards/) — 待补充
- [附录 C 术语表](/appendix/appendix-c-glossary/) — 待补充
- [附录 D 参考资源](/appendix/appendix-d-references/) — 待补充
- [附录 E 社区共建](/appendix/appendix-e-community/) — 待补充
- [附录 F 常见失败模式与排查清单](/appendix/appendix-f-troubleshooting/) — 待补充
