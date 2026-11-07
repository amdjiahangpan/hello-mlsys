---
layout: home
title: Hello MLSys
description: 用模拟器预测驱动 LLM 推理系统设计，在 AMD GPU 上实测验证与校准。
hero:
  name: "Hello MLSys"
  text: "用模拟指导 LLM 推理系统设计"
  tagline: 先用模拟器预测，再上 AMD 真机验证、解释误差、校准模型，最后做出设计决策
  actions:
    - theme: brand
      text: 开始学习
      link: /part0-intro/chapter0/
    - theme: alt
      text: 全书目录
      link: /curriculum/
---

::: info Alpha 内测版
教程正在持续完善，部分章节仍为大纲或待补充内容。阅读时请以各章现有正文与验证记录为准，欢迎通过 GitHub Issue 反馈问题。
:::

## 这本教程学什么

模型能跑起来之后，如何估算显存、预测推理速度、选择量化方案，并判断服务能承受多少并发？Hello MLSys 从这些问题出发，用模拟器筛选方案，再通过 AMD 真机测量解释误差、校准模型，最后做出设计决策。

**预测 → 实测 → 校准 → 留出验证 → 设计决策**

## 建议阅读顺序

1. [入门：协议与纸面演练](/part0-intro/chapter0/)：了解教程定位、测量协议与实验底稿。
2. [建模与校准](/part1-modeling/chapter1/)：从 Roofline 开始，逐步学习显存、量化、通信、并行与服务性能。
3. [与 AI 一起设计](/part2-ai-design/chapter9-1/)：把前面的预测与验证方法用于部署、容量规划与集群设计。
4. [选修与案例库](/part3-electives/)：按需继续探索，相关术语与资料可从[全书目录](/curriculum/)进入。

## 开始之前

建议具备 Python、PyTorch 和 Linux 基础。环境安装可参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm)，单卡算子与 kernel 优化可参考 [hello-gpu](https://github.com/datawhalechina/hello-gpu)。
