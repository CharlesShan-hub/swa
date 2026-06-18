"""
批量评估所有传统 ML 模型，安静运行，输出简洁
"""

import subprocess
import os
import sys
from pathlib import Path
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 定义要评估的模型顺序和对应的算法名称
ALGORITHMS = [
    "linear_model",
    "quadratic_model",
    "random_forest_model",
    "extra_trees_model", 
    "svr_model",
    "xgboost_model",
    "lightgbm_model",
    "catboost_model",
]


def evaluate_model(name: str, data_path: str, output_dir: str) -> None:
    """
    运行单个模型的评估，输出实时显示，同时保存到文件。
    """
    output_file = os.path.join(output_dir, f"{name}.txt")
    
    # stdout 写入文件，stderr 直接透传到终端（tqdm 进度条实时显示）
    with open(output_file, "w", encoding="utf-8") as f:
        subprocess.run(
            ["uv", "run", "python", "scripts/evaluate_by_voltage.py", "--algorithm", name, "--data", data_path],
            stdout=f,
            cwd=os.path.dirname(__file__) + "/..",
        )
    
    # 读取结果文件，提取关键指标打印到终端
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if any(k in line for k in ["MAE:", "RMSE:", "95%分位:", "99%分位:",
                                        "符号准确率:", "投/退判断准确率:", "±5%", "±10%", "±15%"]):
                print(f"  {line.strip()}")


def main():
    parser = argparse.ArgumentParser(description="批量评估所有传统 ML 模型")
    parser.add_argument("--data", default="data/exported_data.jsonl",
                        help="数据文件路径 (默认: data/exported_data.jsonl)")
    parser.add_argument("--output", 
                        help="结果保存目录 (默认: 数据集同名目录下的 evaluation_results)")
    args = parser.parse_args()
    
    # 如果没有指定输出目录，就放在数据集同名目录下
    if args.output is None:
        filename = os.path.basename(args.data)
        name_without_ext = os.path.splitext(filename)[0]
        data_dir = os.path.dirname(args.data) or "data"
        args.output = os.path.join(data_dir, name_without_ext, "evaluation_results")

    os.makedirs(args.output, exist_ok=True)

    print("="*70)
    print("BATCH EVALUATION: TRADITIONAL ML MODELS")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print("="*70, flush=True)
    
    for name in ALGORITHMS:
        evaluate_model(name, args.data, args.output)
    
    print(f"\n{'='*70}")
    print(f"DONE! All results saved to: {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
