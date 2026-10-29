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

## 教学目标

- 能写出 Roofline 模型 Attainable FLOPS = min(Peak FLOPS, AI × Peak Bandwidth) 的含义与适用边界；
- 能计算算子的算术强度，判断它是 Compute-bound 还是 Memory-bound；
- 能用模拟器预测矩阵乘、LayerNorm、FlashAttention 在目标卡上的瓶颈类型与时间量级；
- 能识别"看起来像 Compute-bound、实际受带宽限制"的算子，并解释直觉错在哪里。

## 内容梗概

### Roofline 模型

理解 Attainable FLOPS = min(Peak FLOPS, AI × Peak Bandwidth) 的含义与适用边界：两条"屋顶"给出理论上限，理论峰值是上限而不是当前 kernel 的实际上限。

### 算术强度与瓶颈判定

计算算子的算术强度（AI），对照 ridge 点判断它落在 Compute-bound 还是 Memory-bound 区域。

### 计算时间量级估算

从工作量与硬件峰值估算执行时间的数量级，建立"离硬件极限还有多远"的直觉。

## 实验设计

### 模拟器实验

给定矩阵乘法、LayerNorm 与 FlashAttention，用模拟器预测它们在 W7900D / 9070 XT 上是 Compute-bound 还是 Memory-bound，给出预测时间。故意选一个"看起来像 compute-bound 但实际带宽受限"的算子，让学生校准直觉。

### 真机实验：无

本章全部在模拟器上完成；真机验证与 Profile 自第 2 章开始。

### 闭环与交付

- **预测（测前留存）**：各算子的瓶颈判定、预测时间与选点理由。
- **实测（选点运行）**：本章无。
- **校准（解释误差）**：本章无；"预测 → 实测"的误差校准在第 2 章完成。
- **决策（提交）**：场景定义（模型配置、硬件、负载）、预测曲线与瓶颈判定说明。

## 本章小结

- 本章目前是 Alpha 阶段的内容梗概、实验设计与教学目标骨架，正式正文会在对应实验跑通后补齐。
- 与本章相关的代码、实验配置与测量记录会放在 `code/part1-modeling/chapter1/`。

## 延伸阅读

- 待补充。