#!/bin/bash
# ============================================================
# BMO 代码库主控脚本
# 本仓库包含三类双层极小极大（BMO）算法：
#   1) 一阶基线：SSGDA / TSGDA-1 / TSGDA-2（bmo_core/solvers.py，论文算法 1-3）
#   2) 一阶改进：HSGDA 超梯度辅助算法（hsgda/hsgda_core.py）
#   3) 二/高阶改进：MS-BMO 二阶随机动量算法（bmo_core/solver_ms.py）
#
# 用法：
#   bash run.sh                        # 安装依赖，完整运行三类算法
#   bash run.sh quick                  # 安装依赖，快速冒烟三类算法
#   bash run.sh full   first_order     # 只运行一阶基线（完整模式）
#   bash run.sh full   hsgda           # 只运行 HSGDA（完整模式）
#   bash run.sh full   msbmo           # 只运行 MS-BMO（完整模式）
#   bash run.sh quick  <算法名>         # 单算法快速冒烟
# ============================================================
set -e
cd "$(dirname "$0")"

MODE="${1:-full}"
TARGET="${2:-all}"

# ---------- 第 0 步：依赖安装 ----------
echo "==> 第 0 步：安装依赖 requirements.txt"
pip install -r requirements.txt

# ---------- 分发到各算法脚本 ----------
run_one () {
    local name="$1"
    case "$name" in
        first_order)
            echo ""
            echo "================ 一阶基线 BMO（SSGDA / TSGDA-1 / TSGDA-2） ================"
            bash run_first_order.sh "$MODE"
            ;;
        hsgda)
            echo ""
            echo "================ 一阶改进 HSGDA（超梯度辅助） ================"
            bash run_hsgda.sh "$MODE"
            ;;
        msbmo)
            echo ""
            echo "================ 二/高阶改进 MS-BMO（二阶随机动量） ================"
            bash run_msbmo.sh "$MODE"
            ;;
        *)
            echo "未知算法: $name（可选 first_order | hsgda | msbmo）"
            exit 1
            ;;
    esac
}

if [ "$TARGET" = "all" ]; then
    run_one first_order
    run_one hsgda
    run_one msbmo
else
    run_one "$TARGET"
fi

echo ""
echo "==> 全部完成。各算法结果分别位于 results/ 与 hsgda/results/。"
