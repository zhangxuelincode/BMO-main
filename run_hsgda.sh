#!/bin/bash
# ============================================================
# 一阶改进算法 HSGDA 一键运行脚本
#   算法：HSGDA（Hypergradient-aided Stochastic Gradient Descent-Ascent）
#   实现：hsgda/hsgda_core.py（纯 numpy 合成 BMO 问题，解析可验证）
#   实验：hsgda/run_hsgda_experiments.py
#     E1 conv：收敛与精度验证（HSGDA vs SSGDA vs TSGDA-1，||grad Phi||^2）
#     E2 tk  ：内层步数 K 与外层迭代数 T 的影响（图 1 协议）
#     E3 m1  ：元集大小 m1 的影响，Gap 对照 1/m1（图 2 协议）
#     E4 eta ：步长与调度的影响（图 3/4 协议）
#
# 用法：
#   bash run_hsgda.sh            # 完整流程（默认）
#   bash run_hsgda.sh quick      # 快速冒烟（仅解析校验 + 调参探针）
# ============================================================
set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE

MODE="${1:-full}"
echo "==> [HSGDA] 运行模式: $MODE"

# ---------- 第 1 步：冒烟测试 ----------
# 解析超梯度 vs 有限差分（rel_err < 1e-4）、鞍点/伴随跟踪收缩性、
# HSGDA 收敛而 SSGDA 存在偏差平台
echo "==> 第 1 步：冒烟测试 hsgda/smoke_test.py"
python hsgda/smoke_test.py

if [ "$MODE" = "quick" ]; then
    # 快速模式：只跑调参探针（各步长下 HSGDA 短程稳定性 + Gap 可见性）
    echo "==> 第 2 步（quick）：调参探针 --exp probe"
    python hsgda/run_hsgda_experiments.py --exp probe
else
    # 完整模式：依次运行 E1–E4 四组实验（conv / tk / m1 / eta）
    echo "==> 第 2 步（full）：全部实验 --exp all"
    python hsgda/run_hsgda_experiments.py --exp all

    # ---------- 第 3 步：汇总绘图（latex/images/hsgda_*.pdf + results/*.png） ----------
    echo "==> 第 3 步：汇总绘图 plot_hsgda_figures.py"
    python hsgda/plot_hsgda_figures.py
fi

echo "==> [HSGDA] 完成，结果已写入 hsgda/results/（hsgda_*.csv/npz）"
