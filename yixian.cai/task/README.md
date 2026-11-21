# IMMC 2027 A 题 任务一/二 建模代码

## 运行方式

```bash
cd /Users/ckx/Desktop/WorkBuddy/immc/code
python3 run_all.py
```

## 文件结构

```
immc/
├── datasets/          # 原始下载数据（5个CSV，Epoch AI）
├── clean/             # 清洗后的输入 + 数据字典
│   ├── data_centers_clean.csv
│   ├── timelines_clean.csv
│   ├── chip_quantities_clean.csv
│   ├── chillers_clean.csv
│   ├── cooling_towers_clean.csv
│   └── data_dictionary.json
├── code/              # 本目录
│   ├── 01_clean_data.py
│   ├── 02_task1_carrying_capacity.py
│   ├── 03_task2_distributed_network.py
│   ├── 04_dataset_based_analysis.py
│   ├── 05_make_paper_figures.py
│   └── run_all.py
├── figures/           # 论文图表（PNG 300dpi + PDF 矢量）
└── outputs/           # 模型结果 JSON
    ├── task1_results.json
    └── task2_results.json
```

## 数据清洗要点（01_clean_data.py）

| 原始问题 | 处理 |
|---|---|
| 数值列是字符串 | 全部 `pd.to_numeric(errors='coerce')` |
| 文本列缺失（Owner/Users 等） | 填 'Unknown' |
| 日期格式不统一 | 转 datetime，提取 year |
| Water use (MGD) 单位 | 转 m³/年（1 MGD=1.3817e6 m³/年） |
| 状态列是长文本 | 归类为 operational/land_clearing/under_construction/other |
| 冷却塔/冷水机大量空列 | 删除全空行，Area 缺失时用长×宽补算 |

**关键发现**：`timelines_clean.csv` 的 `water_MGD` 有 84 条真实现场耗水记录，反推 `w_per_IT` 中位数 0.20 m³/MWh——这是**仅现场冷却**口径，与 UNU 报告含发电的 9.84 不同，不可混用。

## 任务一模型（02）

- 三类约束折算成 IT 功率上限（MW），逐年取 min
- 能源：电网剩余 + 自建燃机（含交付延迟）
- 水：min(物理可用水, 政府批水量) / (H_eff × w_per_IT)
- 土地：可用土地 / 单位占地
- 结果：南京=能源瓶颈(304→804MW)，西北=水瓶颈(27→22MW)

## 任务二模型（03）

- 混合整数规划（PuLP/CBC）：启用节点二元变量 + 任务分配连续变量
- 目标：最小化 成本 + 网络 + 环境足迹
- 约束：需求守恒、节点容量、水承载上限、预算、时延
- 鲁棒性：需求+20% / 预算-30% 场景
- 结果：7/12 节点启用，300K H100e 需求满足，鲁棒性通过

## 论文图表（05_make_paper_figures.py）

把 `outputs/dataset_based_results.json` 做成 4 张论文级图表，输出到 `figures/`，每张 PNG(300dpi) + PDF(矢量)：

- **图1 规模分布**：96 站点功率直方图，标注中位数 118 MW 与最大值 1037 MW
- **图2 耗水强度差异**（核心）：15 条真实记录，对数轴，极差 755 倍
- **图3 中国站点规模**：5 个中国站点 vs 美国最大站点（1037 MW）参考线
- **图4 水→算力上限**：可用水量与可支撑功率/算力双轴图

中文用 PingFang 字体，学术克制配色。可单独运行：`python3 05_make_paper_figures.py`

## 依赖

```
pip install pandas numpy pulp==2.7.0 matplotlib
```

注意：
- pulp 需 2.x/3.x 稳定版，4.3.1 的 Rust 后端 API 不兼容传统写法。本环境已装 pulp 2.7.0。
- `run_all.py` 会自动寻找带全部依赖的 Python 解释器，用户终端失效的 .venv 不影响使用。
