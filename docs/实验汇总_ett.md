# ETT 实验结果汇总

---

## 实验协议

| 项 | 值 |
| --- | --- |
| 数据 | ETTh1、ETTm1 |
| 输入模式 | ets_a/m、ets_b、ets_c、ets_h、DLinear：features_only(7ch)；LSTM/GRU/tcn|
| seq_len / pred_len | 96 / 96 |
| 指标 | 归一化空间 Test MSE / MAE |
| 模型 | LSTM、GRU、TCN、DLinear、ets_a、ets_m、ets_b、ets_c、ets_h |

---

## 数据集

| 数据集 | 文件 | 采样 |
| --- | --- | --- |
| ETTh1 | ETTh1.csv | 1 小时 |
| ETTm1 | ETTm1.csv | 15 分钟 |

## ETTh1 · 对比实验

| Rank | 模型 | Params | Test MSE | Test MAE |
| --- | --- | --- | --- | --- |
| 1 | dlinear | 19K | 0.0694 | 0.1972 |
| 2 | ets_a | 23K | 0.1663 | 0.3418 |
| 3 | ets_b | 28K | 0.2785 | 0.4583 |
| 4 | ets_m | 57K | 0.2802 | 0.4533 |
| 5 | gru | 164K | 0.3464 | 0.5350 |
| 6 | ets_c | 9K | 0.3933 | 0.5736 |
| 7 | tcn | 612K | 0.4083 | 0.5865 |
| 8 | ets_h | 318K | 0.5199 | 0.6470 |
| 9 | lstm | 215K | 0.5477 | 0.6920 |

## ETTm1 · 对比实验

| Rank | 模型 | Params | Test MSE | Test MAE |
| --- | --- | --- | --- | --- |
| 1 | dlinear | 19K | 0.0371 | 0.1438 |
| 2 | ets_b | 28K | 0.0694 | 0.2073 |
| 3 | lstm | 215K | 0.0728 | 0.2134 |
| 4 | ets_m | 57K | 0.0764 | 0.2211 |
| 5 | ets_a | 23K | 0.0775 | 0.2234 |
| 6 | ets_c | 9K | 0.0918 | 0.2485 |
| 7 | tcn | 612K | 0.1514 | 0.3364 |
| 8 | gru | 164K | 0.1547 | 0.3380 |
| 9 | ets_h | 318K | 0.7170 | 0.7137 |
