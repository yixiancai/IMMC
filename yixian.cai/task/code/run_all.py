# -*- coding: utf-8 -*-
"""
一键运行：数据清洗 -> 任务一 -> 任务二 -> 数据集分析

自动寻找带 pandas 的 Python 解释器，绕过用户环境中失效的 .venv。
用法：python3 run_all.py   （用任何 python3 调用均可）
     或 bash run.sh
"""
import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# 候选解释器：优先当前，其次已知带依赖的venv，再试系统
CANDIDATES = [
    sys.executable,
    "/Users/ckx/Desktop/git/python/.venv/bin/python3",
    "/Users/ckx/.workbuddy/binaries/python/versions/3.13.12/bin/python3",
    "/usr/bin/python3",
]


def find_python():
    """返回第一个能 import pandas 的解释器"""
    for p in CANDIDATES:
        if not os.path.exists(p):
            continue
        try:
            r = subprocess.run(
                [p, "-c", "import pandas, numpy, pulp, matplotlib"],
                capture_output=True, timeout=30)
            if r.returncode == 0:
                return p
        except Exception:
            continue
    return None


def bootstrap(py):
    """若候选解释器都缺依赖，尝试用 pip 安装"""
    for p in CANDIDATES:
        if not os.path.exists(p):
            continue
        try:
            r = subprocess.run(
                [p, "-m", "pip", "install", "pandas", "numpy", "pulp==2.7.0", "matplotlib", "--quiet"],
                capture_output=True, timeout=300)
            if r.returncode == 0:
                chk = subprocess.run(
                    [p, "-c", "import pandas, numpy, pulp, matplotlib"],
                    capture_output=True, timeout=30)
                if chk.returncode == 0:
                    return p
        except Exception:
            continue
    return None


def main():
    py = find_python()
    if py is None:
        print("正在安装依赖（pandas/numpy/pulp）...")
        py = bootstrap(py if py else CANDIDATES[1])
    if py is None:
        print("!! 无法找到或安装带 pandas/numpy/pulp 的 Python 解释器")
        sys.exit(1)

    print(f"使用解释器: {py}")
    scripts = [
        "01_clean_data.py",
        "02_task1_carrying_capacity.py",
        "03_task2_distributed_network.py",
        "04_dataset_based_analysis.py",
        "05_make_paper_figures.py",
        "07_task2_full_model.py",
        "08_task2_with_real_data.py",
    ]
    for s in scripts:
        print("\n" + "#" * 70)
        print("# 运行", s)
        print("#" * 70)
        r = subprocess.run([py, os.path.join(BASE, s)])
        if r.returncode != 0:
            print(f"!! {s} 运行失败，退出码 {r.returncode}")
            sys.exit(r.returncode)

    print("\n全部完成。结果见 ../outputs/")


if __name__ == "__main__":
    main()
