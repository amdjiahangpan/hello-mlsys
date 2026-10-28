---
title: "第 1 章 模拟器估算、Roofline 与瓶颈"
description: "Hello MLSys · 第 1 章 模拟器估算、Roofline 与瓶颈 · Roofline、算术强度、Compute-bound / Memory-bound 判定、计算时间量级估算"
---

# 第 1 章 模拟器估算、Roofline 与瓶颈

::: danger ⚠️ Alpha内测版本警告
此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。
:::

## 本章导读

> 本章建立性能预测的第一层直觉：一个算子在给定硬件上的理论上限是多少，瓶颈落在计算还是访存。读完后，你应该能先用模拟器给出预测，再带着预测去看真机数据。

## 1.1 Roofline 模型

理解 Attainable FLOPS = min(Peak FLOPS, AI × Peak Bandwidth) 的含义与适用边界。

## 1.2 算术强度与瓶颈判定

计算算子的算术强度，判断它是 Compute-bound 还是 Memory-bound。

## 1.3 计算时间量级估算

从工作量与硬件峰值估算执行时间的数量级。

## 1.4 模拟器实验：预测典型算子的瓶颈与时间

用模拟器预测矩阵乘、LayerNorm 与 FlashAttention 的瓶颈类型和预测时间，并用一个“看起来像 Compute-bound、实际受带宽限制”的算子校准直觉。

## 本章小结

- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。
- 与本章相关的代码、实验配置与测量记录会放在 `code/part1-modeling/chapter1/`。

## 延伸阅读

- 待补充。
