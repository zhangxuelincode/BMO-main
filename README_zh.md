<div align="center">

[English](README.md) | 简体中文

# BMO：双层极小极大优化——基线算法与改进算法

本仓库实现并验证面向鲁棒数据（去噪 / 生成）的三类双层极小极大优化（BMO）算法：
论文中的**一阶基线算法**、**一阶改进的超梯度辅助算法（HSGDA）**，以及
**二/高阶随机动量算法（MS-BMO）**。每类算法都配有独立的一键运行脚本、实验入口与测试。

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2604.20115-red.svg)](https://arxiv.org/abs/2604.20115)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-17%20passed-brightgreen.svg)](tests/test_bmo.py)

</div>

## 📰 新闻

- **[2026-09]** 三类算法各自配齐一键脚本：一阶基线（`run_first_order.sh`）、一阶改进
  **HSGDA**（`run_hsgda.sh`）、二/高阶改进 **MS-BMO**（`run_msbmo.sh`）；主控 `run.sh`
  统一分发。
- **[2026-09]** 以 `bmo_core` 包的形式重新实现了论文中的三个一阶求解器
  （**SSGDA**、**TSGDA-1**、**TSGDA-2**），并通过单元测试（17/17 通过）；发布了复现论文
  图 1–4 趋势的仿真实验。

## ✨ 概述

我们研究面向鲁棒数据（去噪 / 生成）的双层极小极大框架：**上层**在一小撮干净元集上
元学习样本权重，**下层**在带噪训练样本上求解 GAN 式极小极大问题。

```text
min_x  E_xi f(x, y*(x), z*(x); xi)
s.t.   (y*(x), z*(x)) ∈ argmin_y argmax_z E_zeta g(x, y, z; zeta)
```

变量约定与全仓库一致：`x` = 上层加权网络 WeightNet，`y` = 生成器 `G`（下层极小），
`z` = 判别器 `D`（下层极大）。

## 🧮 算法一览

| 类别 | 算法 | 实现 | 关键机制 | 入口脚本 |
|---|---|---|---|---|
| 一阶基线 | **SSGDA**（算法 1） | `bmo_core/solvers.py` | 单时间尺度，每个外层步做一次交替 y/z 更新，步长按外层索引 | `run_first_order.sh` |
| 一阶基线 | **TSGDA-1**（算法 2） | `bmo_core/solvers.py` | 内循环 `K` 步交替更新 y/z，步长按内层索引 | `run_first_order.sh` |
| 一阶基线 | **TSGDA-2**（算法 3） | `bmo_core/solvers.py` | 内循环 `K` 步 y 更新 + `Q` 步 z 更新 | `run_first_order.sh` |
| 一阶改进 | **HSGDA** | `hsgda/hsgda_core.py` | 超梯度辅助的 SSGDA：在解析可验证的合成问题上做鞍点跟踪与伴随跟踪，具有可证的收缩性 | `run_hsgda.sh` |
| 二/高阶改进 | **MS-BMO** | `bmo_core/solver_ms.py` | 下层 STORM 同样本校正动量；鞍点逆估计用新鲜 Hessian oracle 配阻尼 CG 求 `M^{-1}P`；上层完整超梯度 oracle 配 STORM 校正；单循环、单时间尺度 | `run_msbmo.sh` |

**一阶基线的工程设计**

- 🚫 **纯一阶算法** —— 不使用超梯度 / 展开的伪网络。上层采用对偶平均动量并配合梯度裁剪，
  支持多种步长衰减调度（`fixed` / `exp` / `inv` / `inv_sqrt`）。
- 📐 **与理论一致的步长索引** —— 条件 `γ^k ≤ c/((k+1)L)` 在 TSGDA-1/2 中按*内层*索引计
  （每个外层步重置），在 SSGDA 中按*外层*步计。
- 📊 **泛化指标** —— 全程记录加权与 *raw*（排除元网络影响）的训练/验证/测试误差，
  以及泛化差距 `GAP = test − val`。
- 🧪 **仿真协议** —— 数据由自然帧加性噪声生成，元集/测试集规模有限，与论文仿真设置一致。

## 🏗️ 项目结构

```text
ref_code/
├── bmo_core/
│   ├── data_generator.py    # 仿真数据：带噪副本、元集/测试集增广
│   ├── networks.py          # ConvGenerator / ConvDiscriminator / WeightNet（1-10-10-1）
│   ├── bmo_task.py          # 双层目标函数、评估、指标历史
│   ├── solvers.py           # 【类别 1】SSGDA / TSGDA-1 / TSGDA-2（一阶基线）
│   └── solver_ms.py         # 【类别 3】MS-BMO（二阶动量 + Hessian/CG）
├── hsgda/
│   ├── hsgda_core.py        # 【类别 2】解析合成 BMO 问题上的 HSGDA 与基线（纯 numpy）
│   ├── run_hsgda_experiments.py  # E1–E4：收敛、K、m1、步长调度实验
│   ├── plot_hsgda_figures.py     # 汇总绘图（PDF/PNG）
│   └── smoke_test.py        # 解析正确性 + 收敛性冒烟测试
├── train_test_BMO.py        # 一阶三求解器的单次训练入口
├── run_experiment.py        # 论文实验 A（图 1/4）与 B（图 2/3）
├── run_expA_split.py        # 实验 A 的单次 (m1, seed) 运行驱动
├── plot_expA.py             # 汇总实验 A 多次运行并绘图
├── run_ms_experiments.py    # MS-BMO 实验：conv / m1 / T / lr / sched
├── test_msbmo_smoke.py      # MS-BMO 冒烟测试（toy 流形数据）
├── tests/test_bmo.py        # 17 个单元测试（更新规则、K=1 等价性、收敛性）
├── run.sh                   # 主控脚本：装依赖 + 运行三类算法
├── run_first_order.sh       # 【类别 1】一键脚本
├── run_hsgda.sh             # 【类别 2】一键脚本
├── run_msbmo.sh             # 【类别 3】一键脚本
├── results/                 # 仿真实验的 CSV + PNG 输出
└── Create_real_data/        # Chaplin 帧数据管线
```

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/zxlml/BMO-main.git BMO
cd BMO/ref_code
pip install -r requirements.txt
```

### 2. 数据

仿真数据基于自然视频帧（卓别林影片帧）。请从
[Chaplin's Frames (MFCGAN)](https://github.com/zhigang-yao/MFCGAN)
下载并放置于 `Create_real_data/Chaplin/frames/` 目录下。HSGDA 实验是纯合成问题
（仅依赖 numpy），不需要外部数据。

### 3. 一键运行

每类算法都有独立脚本，均接受一个模式参数（默认 `full` 完整模式，`quick` 为小规模冒烟）：

```bash
# 主控脚本：安装依赖后依次运行三类算法（完整或快速）
bash run.sh                 # 完整
bash run.sh quick           # 全部快速冒烟

# 也可以只运行某一类算法
bash run.sh full first_order
bash run.sh full hsgda
bash run.sh full msbmo

# 或直接调用某一类算法的脚本
bash run_first_order.sh           # SSGDA / TSGDA-1 / TSGDA-2：测试 + 训练 + 实验 A/B
bash run_hsgda.sh                 # HSGDA：冒烟测试 + E1–E4 + 绘图
bash run_msbmo.sh                 # MS-BMO：冒烟测试 + conv/m1/T/lr/sched
bash run_hsgda.sh quick           # 任意脚本的小规模版本
```

### 4. 手动训练（一阶求解器）

```bash
# TSGDA-1（默认），实验 A 配置
python train_test_BMO.py --solver TSGDA-1 --T 5 --K 300 --img_size 32 \
    --width_g 64 --width_d 4 --gamma1 0.003 --gamma_schedule fixed \
    --eta 0.005 --eta_schedule exp --eta_decay 0.95

# SSGDA
python train_test_BMO.py --solver SSGDA --T 600 --gamma1 0.01

# TSGDA-2
python train_test_BMO.py --solver TSGDA-2 --T 5 --K 100 --Q 50
```

### 5. 单元测试

```bash
python -m pytest tests/ -q   # 17 通过
```

## 📊 实验

所有输出（CSV + PNG）写入 `results/`（一阶基线与 MS-BMO）与 `hsgda/results/`（HSGDA）。

**类别 1 —— 一阶基线**（`run_experiment.py`）：

```bash
# 实验 A（图 1/4）：TSGDA-1，GAP 与误差随内层迭代 k 的变化（不同元集大小 m1）
python run_experiment.py --exp A --img_size 32 --T 5 --repeats 2

# 实验 B（图 2/3）：SSGDA，GAP 与误差随外层迭代 T 的变化（不同步长调度）
python run_experiment.py --exp B --img_size 32 --T 600 --gamma1 0.01

# 实验 A 的大规模运行可拆分为单次 (m1, seed) 任务：
python run_expA_split.py --m1 500 --seed 100
python plot_expA.py
```

**类别 2 —— HSGDA**（`hsgda/run_hsgda_experiments.py`）：

```bash
python hsgda/run_hsgda_experiments.py --exp probe   # 快速调参探针
python hsgda/run_hsgda_experiments.py --exp conv    # E1：与 SSGDA/TSGDA-1 的收敛对比
python hsgda/run_hsgda_experiments.py --exp tk      # E2：内层 K 与外层 T（图 1 协议）
python hsgda/run_hsgda_experiments.py --exp m1      # E3：元集大小 m1 对照 1/m1（图 2）
python hsgda/run_hsgda_experiments.py --exp eta     # E4：步长与调度（图 3/4）
python hsgda/run_hsgda_experiments.py --exp all     # E1–E4
python hsgda/plot_hsgda_figures.py                  # 汇总绘图
```

**类别 3 —— MS-BMO**（`run_ms_experiments.py`）：

```bash
python run_ms_experiments.py --exp conv   # 收敛验证：MS-BMO vs SSGDA vs TSGDA-1
python run_ms_experiments.py --exp m1     # 元集大小（图 1 协议）
python run_ms_experiments.py --exp T      # 外层迭代数的影响
python run_ms_experiments.py --exp lr     # 学习率 eta（图 3 协议）
python run_ms_experiments.py --exp sched  # 步长调度（图 4 协议）
python run_ms_experiments.py --exp all    # 全部实验（加 --quick 可冒烟）
```

## 📖 参考文献

如果您觉得本仓库对您有帮助，也请考虑引用本实现所基于的以下工作。

<details open>
<summary><b>文献列表</b></summary>

> [1] Meta-Weight-Net: Learning an Explicit Mapping for Sample Weighting (NeurIPS 2019)

```bibtex
@article{shu2019meta,
  title={Meta-weight-net: Learning an explicit mapping for sample weighting},
  author={Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
```

> [2] Manifold Fitting with CycleGAN (PNAS 2024)

```bibtex
@article{yao2024manifold,
  title={Manifold fitting with CycleGAN},
  author={Yao, Zhigang and Su, Jiaji and Yau, Shing-Tung},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={5},
  pages={e2311436121},
  year={2024},
  publisher={National Academy of Sciences}
}
```

</details>

## 🙏 致谢

- [First_Order_BMO](https://github.com/) —— 一阶双层优化工程设计（对偶平均、梯度裁剪）；
- [SiPBA](https://github.com/qichaosustech/SiPBA) —— 一阶双层求解器参考实现；
- [Meta-Weight-Net](https://github.com/xjtushujun/Meta-weight-net) —— 元样本加权框架；
- [MFCGAN](https://github.com/zhigang-yao/MFCGAN) —— Chaplin 帧数据来源。

## 📄 许可证

本项目基于 [MIT License](LICENSE) 发布。

## 🔖 引用本文

如果您觉得本工作对您有帮助，请引用：

```bibtex
@article{zhang2026stability,
  title={On the Stability and Generalization of First-order Bilevel Minimax Optimization},
  author={Zhang, Xuelin and Yuan, Peipei},
  journal={arXiv preprint arXiv:2604.20115},
  year={2026},
  url={https://arxiv.org/abs/2604.20115}
}
```
