#!/bin/bash
# ============================================================
# 二/高阶改进算法 MS-BMO 一键运行脚本
#   算法：MS-BMO（二阶随机动量双层极小极大算法，new2.tex 算法 1）
#   实现：bmo_core/solver_ms.py
#     - 下层 STORM 同样本校正动量（Cutkosky-Orabona 形式）
#     - 鞍点逆估计：新鲜 Hessian oracle + 阻尼 CG 求 M^{-1}P
#     - 上层完整超梯度 oracle + STORM 校正动量
#   实验：run_ms_experiments.py（new2.tex 第 7 节）
#     conv：收敛验证（MS-BMO vs SSGDA vs TSGDA-1，超梯度范数）
#     m1  ：元集大小影响（图 1 协议，Gap = test − val）
#     T   ：外层迭代数影响（过拟合与 Gap 增长）
#     lr  ：学习率 eta 影响（图 3 协议）
#     sched：步长调度影响（Fixed / 0.95 / 0.85，图 4 协议）
#
# 用法：
#   bash run_msbmo.sh            # 完整流程（默认，耗时长）
#   bash run_msbmo.sh quick      # 快速冒烟（T=30、repeats=1）
# ============================================================
set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE

MODE="${1:-full}"
echo "==> [MS-BMO] 运行模式: $MODE"

# ---------- 第 1 步：冒烟测试（toy 流形数据：可运行、超梯度有界、损失下降） ----------
echo "==> 第 1 步：冒烟测试 test_msbmo_smoke.py"
python test_msbmo_smoke.py

# ---------- 第 2 步：实验（conv / m1 / T / lr / sched） ----------
if [ "$MODE" = "quick" ]; then
    echo "==> 第 2 步（quick）：全部实验的小规模版本 --exp all --quick"
    python run_ms_experiments.py --exp all --quick
else
    echo "==> 第 2 步（full）：全部实验 --exp all（conv, m1, T, lr, sched）"
    python run_ms_experiments.py --exp all
fi

echo "==> [MS-BMO] 完成，结果已写入 results/（msbmo_*.csv），图件写入 latex/images/ 与 results/"
