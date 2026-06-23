# ETT 实验结果汇总

---

## 实验协议

| 项 | 值 |
| --- | --- |
| 数据 | ETTh1、ETTm1 |
| 输入模式 | ets_a/m、ets_b、ets_c、ets_h、DLinear：features_only(7ch)；LSTM/GRU/tcn：历史基准（旧架构数字） |
| seq_len / pred_len | 96 / 96 |
| 指标 | 归一化空间 Test MSE / MAE |
| 模型 | LSTM、GRU、TCN、DLinear、ets_a、ets_m、ets_b、ets_c、ets_h |
| 设备 | GPU only |

> 单表按 Test MSE 混排。LSTM/GRU/TCN 为历史基准（未重跑），与其余模型排名仅供参考。

---

## 数据集

| 数据集 | 文件 | 采样 |
| --- | --- | --- |
| ETTh1 | ETTh1.csv | 1 小时 |
| ETTm1 | ETTm1.csv | 15 分钟 |

## ETTh1 · 对比实验

| Rank | 模型 | Params | Test MSE | Test MAE | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | dlinear | 19K | 0.0694 | 0.1972 | DLinear · features_only |
| 2 | ets_c | 23K | 0.1663 | 0.3418 | decomp+TCN · features_only |
| 3 | ets_b | 28K | 0.2785 | 0.4583 | last-step gated TCN · features_only |
| 4 | ets_m | 57K | 0.2802 | 0.4533 | Indiv-TCN decomp · features_only |
| 5 | gru | 164K | 0.3464 | 0.5350 | 历史基准 · 旧架构 |
| 6 | ets_a | 9K | 0.3933 | 0.5736 | Shared-TCN decomp · features_only |
| 7 | tcn | 612K | 0.4083 | 0.5865 | 历史基准 · 旧架构 |
| 8 | ets_h | 318K | 0.5199 | 0.6470 | OmniFusion · features_only |
| 9 | lstm | 215K | 0.5477 | 0.6920 | 历史基准 · 旧架构 |

## ETTm1 · 对比实验

| Rank | 模型 | Params | Test MSE | Test MAE | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1 | dlinear | 19K | 0.0371 | 0.1438 | DLinear · features_only |
| 2 | ets_b | 28K | 0.0694 | 0.2073 | last-step gated TCN · features_only |
| 3 | lstm | 215K | 0.0728 | 0.2134 | 历史基准 · 旧架构 |
| 4 | ets_m | 57K | 0.0764 | 0.2211 | Indiv-TCN decomp · features_only |
| 5 | ets_c | 23K | 0.0775 | 0.2234 | decomp+TCN · features_only |
| 6 | ets_a | 9K | 0.0918 | 0.2485 | Shared-TCN decomp · features_only |
| 7 | tcn | 612K | 0.1514 | 0.3364 | 历史基准 · 旧架构 |
| 8 | gru | 164K | 0.1547 | 0.3380 | 历史基准 · 旧架构 |
| 9 | ets_h | 318K | 0.7170 | 0.7137 | OmniFusion · features_only |

---

## 全局结论

- **DLinear**（~19K，全通道 decomp + readout）在 ETTh1/ETTm1 上均为榜首（MSE 0.069 / 0.037），与 ets_b/c 等同为 features_only(7ch) 协议
- **ets_b / ets_c** 已重构为 mode-agnostic；ETTm1 上 ets_b(0.069) 与 ets_c(0.078) 紧随 DLinear，处于 features_only 组前列
- **ets_m**（~57K，Individual 双分支）为精度对照；ETTm1 MSE 0.076，接近 ets_b/c，但仍弱于 DLinear
- **ets_a**（~9K，Shared 双分支 TCN decomp）为轻量通用线；ETTm1 MSE 0.092，弱于 ets_b/c 与 DLinear
- **LSTM / GRU / TCN** 数字为历史实验（旧架构），与其余模型排名仅供参考；当前代码已对齐 NeuralForecast encoder+MLP 结构，需重跑后更新

> 架构图：[ets_a](ets_a_architecture.svg)、[ets_b](ets_b_architecture.svg)、[ets_c](ets_c_architecture.svg)、[ets_h](ets_h_architecture.svg)、[tcn](tcn_architecture.svg)、[ets_t](ets_t_architecture.svg)
