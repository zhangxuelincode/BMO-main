#!/bin/bash
# ============================================================
# 一阶 BMO 基线算法一键运行脚本
#   算法：SSGDA（论文算法1）、TSGDA-1（算法2）、TSGDA-2（算法3）
#   实现：bmo_core/solvers.py，训练入口 train_test_BMO.py，
#         实验入口 run_experiment.py（实验 A：图1/4；实验 B：图2/3）
#
# 用法：
#   bash run_first_order.sh          # 完整流程（默认）
#   bash run_first_order.sh quick    # 快速冒烟（小规模配置）
# ============================================================
set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE   # 规避 Windows 下 OpenMP 运行时重复加载

MODE="${1:-full}"
echo "==> [一阶 BMO] 运行模式: $MODE"

# ---------- 第 1 步：单元测试（17 个用例：更新规则、K=1 等价性、收敛性） ----------
echo "==> 第 1 步：单元测试 tests/test_bmo.py"
python -m pytest tests/test_bmo.py -q

# ---------- 第 2 步：三个求解器各跑一次单次训练 ----------
echo "==> 第 2 步：单次训练演示（SSGDA / TSGDA-1 / TSGDA-2）"
if [ "$MODE" = "quick" ]; then
    # 快速模式：小 T、小元集，只验证流程可通
    python train_test_BMO.py --solver SSGDA   --T 30 --m1 200 --eval_every 10
    python train_test_BMO.py --solver TSGDA-1 --T 2  --K 50 --eval_ks 10 50 --m1 200
    python train_test_BMO.py --solver TSGDA-2 --T 2  --K 50 --Q 30 --eval_ks 10 50 --m1 200
else
    # 完整模式：与论文实验一致的配置
    python train_test_BMO.py --solver SSGDA   --T 600 --gamma1 0.01
    python train_test_BMO.py --solver TSGDA-1 --T 5 --K 300 --img_size 32 \
        --width_g 64 --width_d 4 --gamma1 0.003 --gamma_schedule fixed \
        --eta 0.005 --eta_schedule exp --eta_decay 0.95
    python train_test_BMO.py --solver TSGDA-2 --T 5 --K 100 --Q 50
fi

# ---------- 第 3 步：实验 A（图 1/4）：TSGDA-1，GAP 与误差随内层迭代 k 的变化 ----------
echo "==> 第 3 步：实验 A（TSGDA-1，内层迭代 k 为横轴，不同元集大小 m1）"
if [ "$MODE" = "quick" ]; then
    python run_experiment.py --exp A --img_size 32 --T 2 --repeats 1 \
        --eval_ks 10 30 50
else
    python run_experiment.py --exp A --img_size 32 --T 5 --repeats 2
fi

# ---------- 第 4 步：实验 B（图 2/3）：SSGDA，GAP 与误差随外层迭代 T 的变化 ----------
echo "==> 第 4 步：实验 B（SSGDA，外层迭代 T 为横轴，不同步长调度）"
if [ "$MODE" = "quick" ]; then
    python run_experiment.py --exp B --img_size 32 --T 60 --repeats 1
else
    python run_experiment.py --exp B --img_size 32 --T 600 --gamma1 0.01
fi

echo "==> [一阶 BMO] 全部完成，结果已写入 results/（expA_*、expB_*）"
