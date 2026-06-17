
"""
分析数据集中各个真值电压的分布情况
"""
import sys
import os
import argparse
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils.loader import load_jsonl


def analyze_data(file_path):
    # 使用现成的工具加载数据
    records = load_jsonl(file_path, extract_features=False)
    print(f"总样本数: {len(records)}")
    
    # 按真值电压分组
    voltage_groups = defaultdict(list)
    for rec in records:
        voltage = rec.get('ACTUAL_VOLTAGE', None)
        if voltage is None:
            continue
        
        # 获取波形数据
        wave_str = rec.get("RTU_REGS_P00_WAVE_DATA", "")
        wave = np.array([float(x) for x in wave_str.split(",")], dtype=np.float64)
        
        # 只保留有完整波形的数据（512点或256点都行）
        if len(wave) >= 128:
            voltage_groups[voltage].append(wave)
    
    print("\n各电压档位样本数:")
    for v in sorted(voltage_groups.keys()):
        print(f"  {v}V: {len(voltage_groups[v])} samples")
    
    # 分析每个电压档位的波形统计
    print("\n各电压档位的波形统计特征:")
    print("  电压档位 | 样本数 |    均值    |   标准差   |   最小值   |   最大值  ")
    print("-" * 70)
    
    all_voltage_info = []
    for v in sorted(voltage_groups.keys()):
        signals = voltage_groups[v]
        # 计算每个样本的波形统计
        means = [np.mean(s) for s in signals]
        stds = [np.std(s) for s in signals]
        
        mean_of_means = np.mean(means)
        std_of_means = np.std(means)
        min_mean = np.min(means)
        max_mean = np.max(means)
        
        all_voltage_info.append({
            'voltage': v,
            'count': len(signals),
            'mean': mean_of_means,
            'std': std_of_means,
            'min': min_mean,
            'max': max_mean
        })
        
        print("  {:8}V | {:6} | {:10.2f} | {:10.2f} | {:10.2f} | {:10.2f}".format(
            int(v), len(signals), mean_of_means, std_of_means, min_mean, max_mean
        ))
    
    return all_voltage_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析数据集中各电压档位的波形分布")
    parser.add_argument("-d", "--data", default="data/exported_data.jsonl",
                        help="数据文件路径 (默认: data/exported_data.jsonl)")
    args = parser.parse_args()

    print("=" * 70)
    print("分析 {}".format(args.data))
    print("=" * 70)
    analyze_data(args.data)
