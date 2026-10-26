# 阶段进度说明（核查日期：2026-10-26）

本文为**简洁的阶段进度**，与运行中的 `results_multi/STATE.json`、`PROGRESS.md`
及各类日志分开，专用于公开仓库的状态记录。仅陈述**已完成并核验**的内容。

## 已完成（已核验）

### 热浪识别（CN05.1）

- 阈值：基准期 1981–2010，逐格点、逐日历日 Tmax P90（5 日滑动窗口、numpy 线性
  插值 Type 7、最小样本 ≥100）；输出 `results_fixed/hw_threshold_1981-2010.nc`。
- 2022 年事件识别（合并定义，见 `docs/heatwave_method_v2.md`）已完成并核验；
  输出 `results_merge/hw_2022_events_merged.csv`、`hw_2022_daily_labels.nc`、
  区域汇总 `hw_2022_region_summary.txt`。
- 1961–2022 年度热浪日数已生成：`results_fixed/hw_annual_1961-2022.nc`。

### 能源分析（ERA5，2022 试点，完整核验）

- 下载 + 完整性核验：2022 暖季小时数据（6 变量，3720 小时，13×37，0 NaN）。
- 小时潜力：`results_hourly/`（Faiman 主方案 + 线性式敏感性）。
- 日聚合：`results_daily/`（日界 A/B 双口径）。
- 分析：`results_monthly/`（热浪 vs 普通月度比较）、`results_complementarity/`
  （Pearson/Spearman 相关）、`results_joint/`（风光同时偏低）、`results_case/`
  （示例格点过程）。

### 多年扩展（2011–2022）

- 已完成“下载 + 完整性核验 + 基础处理”的年份（共 8 年）：
  **2011、2012、2013、2014、2015、2016、2017、2022（复用试点）**。
- 每一年均通过：解包、`verify`（3720 小时 / 6 变量 / 13×37 / 0 NaN / 值域）、
  小时潜力、日聚合（A/B）、标签、对齐。

## 进行中（未完成）

- **2018–2021 年**：下载/处理进行中（2026-10-26 时 2018 年 5–8 月已下载，9–10 月
  及 2019–2021 待处理）。编排进程与看门狗仍在运行，进度见本地
  `results_multi/STATE.json` 与 `orchestrator.log`。
- **多年汇总分析**：合并相关（`era5_wind_solar_correlation_multi.py`）与跨年摘要
  （`era5_multi_year_summary.py`）须待 12 年逐年分析完成后运行，**尚未开始**。

## 诚实报告约定

- 在 2011–2022 完整 12 年分析完成前，**不**生成完整 12 年的汇总或结论，不把
  8 年的“下载+核验+基础处理”表述为完整分析。
- 当前可引用的科学结论仅限 **2022 单年试点**（描述性、非因果），边界见
  `README.md` §8。

## 公开范围

- 已公开：方法文档（`docs/`）、代码（`scripts/`，不含内嵌本机路径的编排驱动）、
  README、本进度说明。
- 未公开：CN05.1 原始数据、ERA5 原始下载、所有派生数值/图件（CN05.1 派生成果
  公开许可未确认）、`results_multi/` 运行状态与日志。
