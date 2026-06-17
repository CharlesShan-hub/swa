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


def evaluate_model(name: str, data_path: str, output_dir: str) -> str:
    """
    安静地评估单个模型，返回结果字符串
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {name:>40}")
    print(f"{'='*70}")
    
    # 运行评估脚本，使用 --algorithm 自动推断模型路径
    result = subprocess.run(
        ["uv", "run", "python", "scripts/evaluate_by_voltage.py", "--algorithm", name, "--data", data_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__) + "/.."
    )
    
    # 保存结果到文件
    output_file = os.path.join(output_dir, f"{name}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
    
    # 打印关键信息
    print(f"Results saved to: {output_file}")
    
    # 从输出中提取并打印整体结果
    for line in result.stdout.split("\n"):
        if "MAE:" in line or "RMSE:" in line or "95%分位:" in line or "99%分位:" in line or \
           "符号准确率:" in line or "投/退判断准确率:" in line:
            print("  " + line.strip())
    
    return result.stdout


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
    print("="*70)
    
    results = {}
    for name in ALGORITHMS:
        results[name] = evaluate_model(name, args.data, args.output)
    
    print("\n" + "="*70)
    print("DONE! All results saved to:", args.output)
    print("="*70)


if __name__ == "__main__":
    main()
